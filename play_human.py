from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from gomoku_zero.config import TrainConfig
from gomoku_zero.gomoku_rules import GomokuRules


REPO_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = REPO_ROOT / "web_human"
DEFAULT_DEVICE = None


@dataclass
class CheckpointInfo:
    path: Path
    config: TrainConfig
    iteration: int | None


@dataclass
class GameSession:
    game_id: str
    checkpoint: Path
    board: np.ndarray
    rules: GomokuRules
    model: Any
    mcts: Any
    n_playout: int
    c_puct: float
    candidate_distance: int | None
    tactical_shortcuts: bool
    human_player: int
    ai_player: int
    current_player: int
    status: str
    winner: int | None
    last_move: int | None
    win_line: list[dict[str, int]] | None
    ai_policy: list[list[float]] | None
    ai_value: float | None
    ai_visits: list[list[int]] | None
    ai_visit_total: int
    ai_selected_policy: float | None
    ai_selected_visits: int | None


SESSIONS: dict[str, GameSession] = {}


def main() -> None:
    args = parse_args()
    server_address = (args.host, args.port)
    handler_class = make_handler(default_device=args.device)
    httpd = ThreadingHTTPServer(server_address, handler_class)
    print(f"Human-vs-AI Gomoku server: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local human-vs-AI Gomoku web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    return parser.parse_args()


def make_handler(default_device: str | None):
    class HumanGameHandler(BaseHTTPRequestHandler):
        server_version = "GomokuHumanServer/1.0"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/checkpoints":
                self.write_json({"checkpoints": list_checkpoints()})
                return
            self.serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/new-game":
                    self.write_json(create_game(self.read_json(), default_device))
                    return
                if parsed.path == "/api/move":
                    self.write_json(play_human_move(self.read_json()))
                    return
                if parsed.path == "/api/debug-predict":
                    self.write_json(debug_predict(self.read_json(), default_device))
                    return
                if parsed.path == "/api/hint":
                    self.write_json(make_hint(self.read_json()))
                    return
                self.write_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint.")
            except Exception as exc:
                self.write_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def serve_static(self, raw_path: str) -> None:
            request_path = unquote(raw_path)
            if request_path in {"", "/"}:
                request_path = "/index.html"
            relative = request_path.lstrip("/")
            path = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT.resolve() not in path.parents and path != STATIC_ROOT.resolve():
                self.write_error(HTTPStatus.FORBIDDEN, "Forbidden.")
                return
            if not path.exists() or not path.is_file():
                self.write_error(HTTPStatus.NOT_FOUND, "Not found.")
                return

            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_error(self, status: HTTPStatus, message: str) -> None:
            self.write_json({"error": message}, status=status)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    return HumanGameHandler


def list_checkpoints() -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for path in sorted(REPO_ROOT.rglob("*.pt")):
        if ".git" in path.parts:
            continue
        try:
            info = read_checkpoint_info(path)
            label = make_checkpoint_label(info)
            checkpoints.append(
                {
                    "id": path.relative_to(REPO_ROOT).as_posix(),
                    "label": label,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "iteration": info.iteration,
                    "boardHeight": info.config.board_height,
                    "boardWidth": info.config.board_width,
                    "channels": info.config.channels,
                    "mctsCandidateDistance": info.config.mcts_candidate_distance,
                    "mctsTacticalShortcuts": info.config.mcts_tactical_shortcuts,
                }
            )
        except Exception:
            checkpoints.append(
                {
                    "id": path.relative_to(REPO_ROOT).as_posix(),
                    "label": str(path.relative_to(REPO_ROOT)),
                    "path": str(path.relative_to(REPO_ROOT)),
                    "iteration": None,
                    "boardHeight": None,
                    "boardWidth": None,
                    "channels": None,
                }
            )
    checkpoints.sort(key=lambda item: (item["boardHeight"] or 0, item["iteration"] or -1, item["path"]))
    return checkpoints


def create_game(payload: dict[str, Any], default_device: str | None) -> dict[str, Any]:
    checkpoint = resolve_checkpoint(payload.get("checkpoint"))
    playouts = int(payload.get("playouts", 200))
    c_puct = float(payload.get("cPuct", 5.0))
    candidate_distance = parse_optional_int(payload.get("candidateDistance"))
    tactical_shortcuts = parse_bool(payload.get("tacticalShortcuts", True))
    human_side = str(payload.get("humanSide", "black"))
    device = resolve_device(str(payload.get("device") or default_device or ""))

    if playouts <= 0:
        raise ValueError("playouts must be positive.")

    info = read_checkpoint_info(checkpoint)
    from gomoku_zero.checkpoint import load_model_checkpoint
    from gomoku_zero.mcts import MCTS

    model = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    rules = GomokuRules(
        board_height=info.config.board_height,
        board_width=info.config.board_width,
        player_values=info.config.player_values,
    )
    first_player, second_player = rules.player_values
    human_player = first_player if human_side == "black" else second_player
    ai_player = rules.opponent_of(human_player)
    mcts = MCTS(
        model=model,
        n_playout=playouts,
        c_puct=c_puct,
        device=device,
        rules=rules,
        candidate_distance=candidate_distance,
        tactical_shortcuts=tactical_shortcuts,
    )
    session = GameSession(
        game_id=uuid.uuid4().hex,
        checkpoint=checkpoint,
        board=np.zeros((rules.board_height, rules.board_width), dtype=np.int8),
        rules=rules,
        model=model,
        mcts=mcts,
        n_playout=playouts,
        c_puct=c_puct,
        candidate_distance=candidate_distance,
        tactical_shortcuts=tactical_shortcuts,
        human_player=human_player,
        ai_player=ai_player,
        current_player=first_player,
        status="playing",
        winner=None,
        last_move=None,
        win_line=None,
        ai_policy=None,
        ai_value=None,
        ai_visits=None,
        ai_visit_total=0,
        ai_selected_policy=None,
        ai_selected_visits=None,
    )
    SESSIONS[session.game_id] = session

    if session.current_player == session.ai_player:
        make_ai_move(session)

    return serialize_session(session)


def make_hint(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", ""))
    if game_id not in SESSIONS:
        raise ValueError("Game not found. Start a new game.")

    session = SESSIONS[game_id]
    if session.status != "playing":
        raise ValueError("The game has ended.")
    if session.current_player != session.human_player:
        raise ValueError("Hint is only available on the human player's turn.")

    from gomoku_zero.mcts import MCTS

    current_player = session.human_player
    policy, value = model_prediction(session, current_player, temperature=1.0)
    hint_mcts = MCTS(
        model=session.model,
        n_playout=session.n_playout,
        c_puct=session.c_puct,
        device=next(session.model.parameters()).device,
        rules=session.rules,
        candidate_distance=session.candidate_distance,
        tactical_shortcuts=session.tactical_shortcuts,
    )
    _, visits, visit_total = select_move_with_visits(
        mcts=hint_mcts,
        board=session.board,
        player=current_player,
        rules=session.rules,
    )
    return {
        "gameId": session.game_id,
        "boardHeight": session.rules.board_height,
        "boardWidth": session.rules.board_width,
        "currentPlayer": current_player,
        "policy": policy,
        "value": value,
        "visits": visits,
        "visitTotal": visit_total,
        "policyTemperature": 1.0,
    }


def debug_predict(payload: dict[str, Any], default_device: str | None) -> dict[str, Any]:
    checkpoint = resolve_checkpoint(payload.get("checkpoint"))
    device = resolve_device(str(payload.get("device") or default_device or ""))
    info = read_checkpoint_info(checkpoint)
    from gomoku_zero.checkpoint import load_model_checkpoint

    rules = GomokuRules(
        board_height=info.config.board_height,
        board_width=info.config.board_width,
        player_values=info.config.player_values,
    )
    board = np.asarray(payload.get("board"), dtype=np.int8)
    board = rules.as_board(board)
    current_player = int(payload.get("currentPlayer", rules.player_values[0]))
    rules.validate_player(current_player)

    model = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    policy, value = model_prediction_for(
        model=model,
        rules=rules,
        board=board,
        current_player=current_player,
        temperature=1.0,
    )
    return {
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "board": board.astype(int).tolist(),
        "boardHeight": rules.board_height,
        "boardWidth": rules.board_width,
        "currentPlayer": current_player,
        "aiPolicy": policy,
        "aiValue": value,
        "policyTemperature": 1.0,
    }


def play_human_move(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", ""))
    row = int(payload.get("row"))
    col = int(payload.get("col"))
    if game_id not in SESSIONS:
        raise ValueError("Game not found. Start a new game.")

    session = SESSIONS[game_id]
    if session.status != "playing":
        return serialize_session(session)
    if session.current_player != session.human_player:
        raise ValueError("It is not the human player's turn.")
    if not (0 <= row < session.rules.board_height and 0 <= col < session.rules.board_width):
        raise ValueError("Move is outside the board.")

    move = row * session.rules.board_width + col
    session.board = session.rules.next_board(session.board, move, session.human_player)
    session.last_move = move
    session.mcts.update_with_move(move)
    update_game_status(session, move, session.human_player)

    if session.status == "playing":
        session.current_player = session.ai_player
        make_ai_move(session)

    return serialize_session(session)


def make_ai_move(session: GameSession) -> None:
    if session.status != "playing":
        return
    session.ai_policy, session.ai_value = model_prediction(session, session.ai_player, temperature=1.0)
    move, session.ai_visits, session.ai_visit_total = select_ai_move_with_visits(session)
    row, col = divmod(move, session.rules.board_width)
    session.ai_selected_policy = float(session.ai_policy[row][col]) if session.ai_policy else None
    session.ai_selected_visits = int(session.ai_visits[row][col]) if session.ai_visits else None
    session.board = session.rules.next_board(session.board, move, session.ai_player)
    session.last_move = move
    session.mcts.update_with_move(move)
    update_game_status(session, move, session.ai_player)
    if session.status == "playing":
        session.current_player = session.human_player


def select_ai_move_with_visits(session: GameSession) -> tuple[int, list[list[int]], int]:
    return select_move_with_visits(
        mcts=session.mcts,
        board=session.board,
        player=session.ai_player,
        rules=session.rules,
    )


def select_move_with_visits(
    mcts: Any,
    board: np.ndarray,
    player: int,
    rules: GomokuRules,
) -> tuple[int, list[list[int]], int]:
    moves, move_probs = mcts.get_action_probs(board, player, temp=1e-3)
    if not moves:
        raise ValueError("No legal moves available.")

    move = int(np.random.choice(moves, p=move_probs))
    visits = np.zeros(rules.board_size, dtype=np.int32)
    for child_move, child in mcts.root.children.items():
        visits[int(child_move)] = int(child.visits)

    total = int(visits.sum())
    if total == 0:
        visits[move] = 1
        total = 1

    return move, visits.reshape(rules.board_height, rules.board_width).tolist(), total


def model_prediction(
    session: GameSession,
    current_player: int,
    temperature: float = 1.0,
) -> tuple[list[list[float]], float]:
    return model_prediction_for(
        model=session.model,
        rules=session.rules,
        board=session.board,
        current_player=current_player,
        temperature=temperature,
    )


def model_prediction_for(
    model: Any,
    rules: GomokuRules,
    board: np.ndarray,
    current_player: int,
    temperature: float = 1.0,
) -> tuple[list[list[float]], float]:
    legal_moves = rules.legal_moves(board)
    policy = np.zeros(rules.board_size, dtype=np.float32)
    if len(legal_moves) == 0:
        return policy.reshape(rules.board_height, rules.board_width).tolist(), 0.5

    state = rules.encode_state(board, current_player)
    import_torch()
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        policy_logits, value = model(tensor)
        logits = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
        value_prediction = float(value.item())

    legal_logits = logits[legal_moves]
    if temperature <= 0:
        best_move = int(legal_moves[int(np.argmax(legal_logits))])
        policy[best_move] = 1.0
    else:
        scaled = legal_logits / float(temperature)
        scaled = scaled - float(np.max(scaled))
        probs = np.exp(scaled)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            probs = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float64)
        else:
            probs = probs / prob_sum
        policy[legal_moves] = probs.astype(np.float32)

    return policy.reshape(rules.board_height, rules.board_width).tolist(), value_prediction


def update_game_status(session: GameSession, move: int, player: int) -> None:
    ended, winner = session.rules.game_end_after_move(session.board, move, player)
    if ended:
        session.status = "ended"
        session.winner = int(winner)
        if winner != 0:
            session.win_line = find_winning_line(session.rules, session.board, move, player)


def find_winning_line(
    rules: GomokuRules,
    board: np.ndarray,
    move: int,
    player: int,
) -> list[dict[str, int]] | None:
    row, col = divmod(int(move), rules.board_width)
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dr, dc in directions:
        cells = [(row, col)]
        r, c = row - dr, col - dc
        while 0 <= r < rules.board_height and 0 <= c < rules.board_width and board[r, c] == player:
            cells.insert(0, (r, c))
            r -= dr
            c -= dc

        r, c = row + dr, col + dc
        while 0 <= r < rules.board_height and 0 <= c < rules.board_width and board[r, c] == player:
            cells.append((r, c))
            r += dr
            c += dc

        if len(cells) >= 5:
            return [{"row": int(cell_row), "col": int(cell_col)} for cell_row, cell_col in cells]
    return None


def serialize_session(session: GameSession) -> dict[str, Any]:
    return {
        "gameId": session.game_id,
        "checkpoint": str(session.checkpoint.relative_to(REPO_ROOT)),
        "board": session.board.astype(int).tolist(),
        "boardHeight": session.rules.board_height,
        "boardWidth": session.rules.board_width,
        "humanPlayer": session.human_player,
        "aiPlayer": session.ai_player,
        "currentPlayer": session.current_player,
        "status": session.status,
        "winner": session.winner,
        "lastMove": None
        if session.last_move is None
        else {
            "row": session.last_move // session.rules.board_width,
            "col": session.last_move % session.rules.board_width,
        },
        "winLine": session.win_line,
        "aiPolicy": session.ai_policy,
        "aiValue": session.ai_value,
        "aiVisits": session.ai_visits,
        "aiVisitTotal": session.ai_visit_total,
        "aiSelectedPolicy": session.ai_selected_policy,
        "aiSelectedVisits": session.ai_selected_visits,
        "policyTemperature": 1.0,
    }


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("candidateDistance must be empty or non-negative.")
    return parsed


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_checkpoint(checkpoint_id: Any) -> Path:
    if not checkpoint_id:
        raise ValueError("checkpoint is required.")
    path = (REPO_ROOT / str(checkpoint_id)).resolve()
    if REPO_ROOT not in path.parents and path != REPO_ROOT:
        raise ValueError("Checkpoint path must stay inside this repository.")
    if path.suffix.lower() != ".pt" or not path.exists() or not path.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint_id}")
    return path


def resolve_device(device: str) -> str:
    if device:
        return device
    torch = import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_checkpoint_info(path: Path) -> CheckpointInfo:
    torch = import_torch()
    from gomoku_zero.checkpoint import config_from_checkpoint_payload

    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu")
    config = config_from_checkpoint_payload(checkpoint, path)
    return CheckpointInfo(path=path, config=config, iteration=checkpoint.get("iteration"))


def make_checkpoint_label(info: CheckpointInfo) -> str:
    relative = info.path.relative_to(REPO_ROOT)
    iteration = "initial" if info.iteration is None else f"iter {info.iteration:04d}"
    board = f"{info.config.board_height}x{info.config.board_width}"
    return f"{iteration} · {board} · {relative}"


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to load checkpoints and play against the AI.") from exc
    return torch


if __name__ == "__main__":
    main()
