from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Move = int


@dataclass(frozen=True)
class GomokuRules:
    """Board utilities and game-end rules for freestyle Gomoku."""

    board_height: int = 15
    board_width: int = 15
    player_values: tuple[int, int] = (1, -1)

    def __init__(
        self,
        board_height: int = 15,
        board_width: int = 15,
        player_values: Sequence[int] = (1, -1),
    ) -> None:
        if len(player_values) != 2 or player_values[0] == player_values[1]:
            raise ValueError("player_values must contain two distinct player encodings.")

        object.__setattr__(self, "board_height", board_height)
        object.__setattr__(self, "board_width", board_width)
        object.__setattr__(self, "player_values", (int(player_values[0]), int(player_values[1])))

    @property
    def board_size(self) -> int:
        return self.board_height * self.board_width

    def as_board(self, board: np.ndarray) -> np.ndarray:
        board = np.asarray(board)
        expected_shape = (self.board_height, self.board_width)
        if board.shape != expected_shape:
            raise ValueError(f"Expected board shape {expected_shape}, got {board.shape}.")
        return board

    def legal_moves(self, board: np.ndarray) -> np.ndarray:
        board = self.as_board(board)
        return np.flatnonzero(board.reshape(-1) == 0)

    def next_board(self, board: np.ndarray, move: Move, current_player: int) -> np.ndarray:
        board = self.as_board(board)
        self.validate_player(current_player)

        row, col = divmod(int(move), self.board_width)
        if not (0 <= row < self.board_height and 0 <= col < self.board_width):
            raise ValueError(f"Illegal move {move}: outside the board.")
        if board[row, col] != 0:
            raise ValueError(f"Illegal move {move}: position is already occupied.")

        next_board = board.copy()
        next_board[row, col] = current_player
        return next_board

    def next_player(self, current_player: int) -> int:
        return self.opponent_of(current_player)

    def opponent_of(self, current_player: int) -> int:
        self.validate_player(current_player)
        if current_player == self.player_values[0]:
            return self.player_values[1]
        return self.player_values[0]

    def validate_player(self, current_player: int) -> None:
        if current_player not in self.player_values:
            raise ValueError(f"current_player must be one of {self.player_values}, got {current_player}.")

    def encode_state(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """Return 2 channels: current player's stones, then opponent's stones."""
        board = self.as_board(board)
        opponent = self.opponent_of(current_player)
        return np.stack(
            [
                board == current_player,
                board == opponent,
            ]
        ).astype(np.float32)

    def game_end(self, board: np.ndarray) -> tuple[bool, int]:
        """Return (ended, winner). winner is 0 for draw or unfinished games."""
        board = self.as_board(board)
        for player in self.player_values:
            if self.has_five(board, player):
                return True, int(player)
        if not np.any(board == 0):
            return True, 0
        return False, 0

    def has_five(self, board: np.ndarray, player: int) -> bool:
        board = self.as_board(board)
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        rows, cols = board.shape

        for row, col in np.argwhere(board == player):
            for dr, dc in directions:
                prev_r, prev_c = row - dr, col - dc
                if 0 <= prev_r < rows and 0 <= prev_c < cols and board[prev_r, prev_c] == player:
                    continue

                count = 0
                r, c = row, col
                while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                    count += 1
                    if count >= 5:
                        return True
                    r += dr
                    c += dc
        return False
