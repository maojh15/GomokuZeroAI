const checkpointSelect = document.querySelector("#checkpointSelect");
const modelName = document.querySelector("#modelName");
const boardEl = document.querySelector("#board");
const lastMoveLabel = document.querySelector("#lastMoveLabel");
const moveCountLabel = document.querySelector("#moveCountLabel");
const policySummary = document.querySelector("#policySummary");
const visitsSummary = document.querySelector("#visitsSummary");
const valueSummary = document.querySelector("#valueSummary");
const selectedMoveSummary = document.querySelector("#selectedMoveSummary");
const hintToggle = document.querySelector("#hintToggle");
const policyToggle = document.querySelector("#policyToggle");
const visitsToggle = document.querySelector("#visitsToggle");
const overlayHelp = document.querySelector("#overlayHelp");
const statusTitle = document.querySelector("#statusTitle");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const settingsBtn = document.querySelector("#settingsBtn");
const settingsDialog = document.querySelector("#settingsDialog");
const settingsCloseBtn = document.querySelector("#settingsCloseBtn");
const newGameBtn = document.querySelector("#newGameBtn");
const exportGameBtn = document.querySelector("#exportGameBtn");
const playoutsInput = document.querySelector("#playoutsInput");
const evalBatchSizeInput = document.querySelector("#evalBatchSizeInput");
const cPuctInput = document.querySelector("#cPuctInput");
const candidateDistanceInput = document.querySelector("#candidateDistanceInput");
const humanReplayPathInput = document.querySelector("#humanReplayPathInput");
const tacticalShortcutsInput = document.querySelector("#tacticalShortcutsInput");
const debugModeInput = document.querySelector("#debugModeInput");
const debugPredictBtn = document.querySelector("#debugPredictBtn");
const paintBlack = document.querySelector("#paintBlack");
const paintWhite = document.querySelector("#paintWhite");
const paintEmpty = document.querySelector("#paintEmpty");
const evalBlack = document.querySelector("#evalBlack");
const evalWhite = document.querySelector("#evalWhite");
const sideBlack = document.querySelector("#sideBlack");
const sideWhite = document.querySelector("#sideWhite");

let checkpoints = [];
let game = null;
let humanSide = "black";
let busy = false;
let showPolicy = false;
let showVisits = false;
let showHint = false;
let hintData = null;
let debugMode = false;
let paintValue = 1;
let debugEvalPlayer = 1;
let preDebugGame = null;

sideBlack.addEventListener("click", () => setHumanSide("black"));
sideWhite.addEventListener("click", () => setHumanSide("white"));
settingsBtn.addEventListener("click", openSettings);
settingsCloseBtn.addEventListener("click", closeSettings);
newGameBtn.addEventListener("click", startGame);
exportGameBtn.addEventListener("click", exportGameData);
hintToggle.addEventListener("click", toggleHint);
debugModeInput.addEventListener("change", () => setDebugMode(debugModeInput.checked));
debugPredictBtn.addEventListener("click", runDebugPredict);
paintBlack.addEventListener("click", () => setPaintValue(1));
paintWhite.addEventListener("click", () => setPaintValue(-1));
paintEmpty.addEventListener("click", () => setPaintValue(0));
evalBlack.addEventListener("click", () => setDebugEvalPlayer(1));
evalWhite.addEventListener("click", () => setDebugEvalPlayer(-1));
policyToggle.addEventListener("click", () => {
  showPolicy = !showPolicy;
  if (showPolicy) showVisits = false;
  updatePolicyToggle();
  updateVisitsToggle();
  renderBoard(game);
});
visitsToggle.addEventListener("click", () => {
  showVisits = !showVisits;
  if (showVisits) showPolicy = false;
  updatePolicyToggle();
  updateVisitsToggle();
  renderBoard(game);
});

loadCheckpoints();
renderBoard(null);
updateHintToggle();
updatePolicyToggle();
updateVisitsToggle();
updateExportButton();

function setHumanSide(side) {
  humanSide = side;
  sideBlack.classList.toggle("active", side === "black");
  sideWhite.classList.toggle("active", side === "white");
}

async function loadCheckpoints() {
  setStatus("waiting", "正在载入 checkpoint...", "扫描仓库里的 .pt 文件");
  try {
    const data = await requestJson("/api/checkpoints");
    checkpoints = data.checkpoints || [];
    checkpointSelect.innerHTML = "";
    for (const checkpoint of checkpoints) {
      const option = document.createElement("option");
      option.value = checkpoint.id;
      option.textContent = checkpoint.label;
      checkpointSelect.appendChild(option);
    }
    if (checkpoints.length > 0) {
      checkpointSelect.selectedIndex = checkpoints.length - 1;
      modelName.textContent = checkpointSelect.selectedOptions[0].textContent;
      applyCheckpointDefaults();
      setStatus("ready", "可以开始", "选择 checkpoint 后开局");
    } else {
      setStatus("ended", "没有找到 checkpoint", "训练后生成 .pt 文件再刷新页面");
    }
  } catch (error) {
    setStatus("ended", "载入失败", error.message);
  }
}

checkpointSelect.addEventListener("change", () => {
  modelName.textContent = checkpointSelect.selectedOptions[0]?.textContent || "未选择";
  applyCheckpointDefaults();
});

async function startGame() {
  if (!checkpointSelect.value) return;
  setBusy(true);
  setStatus("waiting", "正在开局", "模型载入和首步搜索可能需要一点时间");
  try {
    game = await requestJson("/api/new-game", {
      checkpoint: checkpointSelect.value,
      humanSide,
      playouts: Number(playoutsInput.value),
      evalBatchSize: Number(evalBatchSizeInput.value),
      cPuct: Number(cPuctInput.value),
      candidateDistance: candidateDistanceInput.value,
      humanReplayPath: humanReplayPathInput.value,
      tacticalShortcuts: tacticalShortcutsInput.checked,
    });
    clearHint();
    modelName.textContent = checkpointSelect.selectedOptions[0].textContent;
    renderGame();
  } catch (error) {
    setStatus("ended", "开局失败", error.message);
  } finally {
    setBusy(false);
    if (game) renderGame();
  }
}

async function makeMove(row, col) {
  if (!game || busy || game.status !== "playing" || game.currentPlayer !== game.humanPlayer) return;
  if (game.board?.[row]?.[col] !== 0) return;

  const previousGame = cloneGame(game);
  const optimisticGame = cloneGame(game);
  optimisticGame.board[row][col] = optimisticGame.humanPlayer;
  optimisticGame.lastMove = { row, col };
  optimisticGame.currentPlayer = optimisticGame.aiPlayer;
  optimisticGame.aiPolicy = null;
  optimisticGame.aiValue = null;
  optimisticGame.aiMctsValue = null;
  optimisticGame.aiVisits = null;
  optimisticGame.aiVisitTotal = 0;
  optimisticGame.aiSelectedPolicy = null;
  optimisticGame.aiSelectedVisits = null;
  optimisticGame.winLine = null;
  clearHint();
  game = optimisticGame;
  renderGame();

  setBusy(true);
  setStatus("waiting", "AI 思考中", "正在运行 MCTS");
  try {
    game = await requestJson("/api/move", {
      gameId: game.gameId,
      row,
      col,
    });
    renderGame();
  } catch (error) {
    game = previousGame;
    renderGame();
    setStatus("ended", "落子失败", error.message);
  } finally {
    setBusy(false);
    if (game) renderGame();
  }
}

function cloneGame(source) {
  return {
    ...source,
    board: source.board.map((boardRow) => [...boardRow]),
    aiPolicy: source.aiPolicy ? source.aiPolicy.map((policyRow) => [...policyRow]) : null,
    aiMctsValue: source.aiMctsValue,
    aiVisits: source.aiVisits ? source.aiVisits.map((visitRow) => [...visitRow]) : null,
    aiSelectedPolicy: source.aiSelectedPolicy,
    aiSelectedVisits: source.aiSelectedVisits,
    mctsBackend: source.mctsBackend,
    mctsBackendNote: source.mctsBackendNote,
    evalBatchSize: source.evalBatchSize,
    humanReplayPath: source.humanReplayPath,
    exported: source.exported,
    lastMove: source.lastMove ? { ...source.lastMove } : null,
    winLine: source.winLine ? source.winLine.map((point) => ({ ...point })) : null,
  };
}

function renderGame() {
  renderBoard(game);
  lastMoveLabel.textContent = makeLastMoveText(game);
  updateExportButton();
  if (debugMode) {
    setStatus("ready", "Debug 摆盘模式", "摆好局面后点击检测网络输出");
  } else if (game.status === "ended") {
    const result = game.winner === 0 ? "平局" : game.winner === game.humanPlayer ? "你赢了" : "AI 获胜";
    setStatus("ended", result, `点击开始新局可重新挑战${backendStatusText(game)}`);
  } else if (game.currentPlayer === game.humanPlayer) {
    setStatus("ready", "轮到你落子", `${stoneName(game.humanPlayer)}${backendStatusText(game)}`);
  } else {
    setStatus("waiting", "等待 AI", `${stoneName(game.aiPlayer)}${backendStatusText(game)}`);
  }
}

function openSettings() {
  if (busy) return;
  if (typeof settingsDialog.showModal === "function") {
    settingsDialog.showModal();
  } else {
    settingsDialog.setAttribute("open", "");
  }
}

function closeSettings() {
  settingsDialog.close();
}

function makeLastMoveText(currentGame) {
  if (debugMode) {
    return currentGame.lastMove
      ? `Debug 最近摆子 ${currentGame.lastMove.row + 1}, ${currentGame.lastMove.col + 1}`
      : "Debug 摆盘";
  }
  return currentGame.lastMove
    ? `最近落子 ${currentGame.lastMove.row + 1}, ${currentGame.lastMove.col + 1}`
    : "等待落子";
}

function renderBoard(currentGame) {
  const size = currentGame?.boardWidth || 15;
  const height = currentGame?.boardHeight || 15;
  const overlay = activeOverlayData(currentGame);
  const policy = overlay.policy;
  const maxPolicy = policy ? Math.max(...policy.flat()) : 0;
  const topMoves = topPolicyMoves(policy);
  const visits = overlay.visits;
  const maxVisits = visits ? Math.max(...visits.flat()) : 0;
  const topVisits = topVisitMoves(visits);
  const winPoints = winLinePointSet(currentGame?.winLine);
  boardEl.style.setProperty("--size", String(size));
  boardEl.innerHTML = "";
  updatePolicySummary(policy, maxPolicy);
  updateVisitsSummary(visits, maxVisits, overlay.visitTotal || 0);
  updateMoveCountSummary(currentGame);
  updateValueSummary(overlay);
  updateSelectedMoveSummary(currentGame);
  updateHintToggle();

  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const value = currentGame?.board?.[row]?.[col] || 0;
      const probability = policy?.[row]?.[col] || 0;
      const intensity = maxPolicy > 0 ? probability / maxPolicy : 0;
      const visitCount = visits?.[row]?.[col] || 0;
      const visitIntensity = maxVisits > 0 ? visitCount / maxVisits : 0;
      const cell = document.createElement("button");
      cell.className = "cell";
      if (row === 0) cell.classList.add("top-edge");
      if (row === height - 1) cell.classList.add("bottom-edge");
      if (col === 0) cell.classList.add("left-edge");
      if (col === size - 1) cell.classList.add("right-edge");
      if (showPolicy && probability > 0) {
        cell.classList.add("policyCell");
        cell.style.setProperty("--policy", String(intensity));
      }
      if (showVisits && visitCount > 0) {
        cell.classList.add("visitCell");
        cell.style.setProperty("--visits", String(visitIntensity));
      }
      cell.type = "button";
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", `${row + 1}, ${col + 1}`);
      cell.title = overlayTitle(row, col, probability, visitCount);
      cell.disabled = value !== 0 || !currentGame || currentGame.status !== "playing" || busy;
      if (debugMode) {
        cell.disabled = busy;
        cell.addEventListener("click", () => paintDebugCell(row, col));
      } else {
        cell.disabled = value !== 0 || !currentGame || currentGame.status !== "playing" || busy;
        cell.addEventListener("click", () => makeMove(row, col));
      }
      if (currentGame?.lastMove?.row === row && currentGame?.lastMove?.col === col) {
        cell.classList.add("last");
      }
      if (winPoints.has(`${row},${col}`)) {
        cell.classList.add("winningCell");
      }
      if (value !== 0) {
        const stone = document.createElement("span");
        stone.className = `stone ${value === 1 ? "black" : "white"}`;
        cell.appendChild(stone);
      } else if (showPolicy && probability > 0 && topMoves.has(`${row},${col}`)) {
        const label = document.createElement("span");
        label.className = "probabilityLabel";
        label.textContent = formatProbability(probability);
        cell.appendChild(label);
      } else if (showVisits && visitCount > 0 && topVisits.has(`${row},${col}`)) {
        const label = document.createElement("span");
        label.className = "visitLabel";
        label.textContent = formatVisitCount(visitCount);
        cell.appendChild(label);
      }
      if (isStarPoint(row, col, height, size)) {
        const star = document.createElement("span");
        star.className = "starPoint";
        cell.appendChild(star);
      }
      boardEl.appendChild(cell);
    }
  }
  renderWinLine(currentGame, height, size);
}

function renderWinLine(currentGame, height, width) {
  const line = currentGame?.winLine;
  if (!line || line.length < 2) return;

  const first = line[0];
  const last = line[line.length - 1];
  const x1 = ((first.col + 0.5) / width) * 100;
  const y1 = ((first.row + 0.5) / height) * 100;
  const x2 = ((last.col + 0.5) / width) * 100;
  const y2 = ((last.row + 0.5) / height) * 100;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const layer = document.createElement("span");
  layer.className = "winLineLayer";
  const segment = document.createElement("span");
  segment.className = "winLineBar";
  segment.style.setProperty("--x", `${x1}%`);
  segment.style.setProperty("--y", `${y1}%`);
  segment.style.setProperty("--length", `${Math.hypot(dx, dy)}%`);
  segment.style.setProperty("--angle", `${Math.atan2(dy, dx)}rad`);
  layer.appendChild(segment);
  boardEl.appendChild(layer);
}

function winLinePointSet(line) {
  if (!line) return new Set();
  return new Set(line.map((point) => `${point.row},${point.col}`));
}

function isStarPoint(row, col, height, width) {
  if (height !== width || height < 9) return false;
  const center = Math.floor(height / 2);
  const near = 3;
  const far = height - 4;
  return (
    (row === center && col === center) ||
    (row === near && col === near) ||
    (row === near && col === far) ||
    (row === far && col === near) ||
    (row === far && col === far)
  );
}

function topPolicyMoves(policy) {
  if (!policy) return new Set();
  return new Set(
    policy
      .flatMap((rowValues, row) => rowValues.map((value, col) => ({ row, col, value })))
      .filter((item) => item.value > 0)
      .sort((a, b) => b.value - a.value)
      .slice(0, 12)
      .map((item) => `${item.row},${item.col}`),
  );
}

function topVisitMoves(visits) {
  if (!visits) return new Set();
  return new Set(
    visits
      .flatMap((rowValues, row) => rowValues.map((value, col) => ({ row, col, value })))
      .filter((item) => item.value > 0)
      .sort((a, b) => b.value - a.value)
      .slice(0, 12)
      .map((item) => `${item.row},${item.col}`),
  );
}

function updatePolicyToggle() {
  policyToggle.classList.toggle("active", showPolicy);
  policyToggle.setAttribute("aria-pressed", String(showPolicy));
  updateOverlayHelp();
}

function updateVisitsToggle() {
  visitsToggle.classList.toggle("active", showVisits);
  visitsToggle.setAttribute("aria-pressed", String(showVisits));
  updateOverlayHelp();
}

function updateOverlayHelp() {
  if (showPolicy) {
    overlayHelp.hidden = false;
    overlayHelp.className = "overlayHelp policyHelp";
    overlayHelp.textContent =
      `${showHint ? "Hint Policy 显示当前玩家视角的提示网络输出。" : "Policy 显示上一手 AI 思考时的网络输出。"}Temperature=1 后的合法落点概率，红色越深表示概率越高，数字标出 Top 12 概率点，标题中的 max 是单点最高概率。`;
  } else if (showVisits) {
    overlayHelp.hidden = false;
    overlayHelp.className = "overlayHelp visitsHelp";
    overlayHelp.textContent =
      `${showHint ? "Hint Visits 显示当前玩家视角的提示搜索结果。" : "Visits 显示上一手 AI 的 MCTS 搜索结果。"}Root 节点各候选落点访问次数，蓝色越深表示访问越多，数字标出 Top 12 访问点，标题中的 total/max 分别是总访问次数和单点最高访问次数。`;
  } else {
    overlayHelp.hidden = true;
    overlayHelp.textContent = "";
  }
}

function updatePolicySummary(policy, maxPolicy) {
  if (!showPolicy) {
    policySummary.textContent = "";
    return;
  }
  policySummary.textContent = policy
    ? `Policy T=1 max ${(maxPolicy * 100).toFixed(2)}%`
    : "等待 policy";
}

function updateVisitsSummary(visits, maxVisits, totalVisits) {
  if (!showVisits) {
    visitsSummary.textContent = "";
    return;
  }
  visitsSummary.textContent = visits
    ? `Visits total ${totalVisits} max ${maxVisits}`
    : "等待 visits";
}

async function exportGameData() {
  if (!game || busy || debugMode || game.status !== "ended" || game.exported) return;

  setBusy(true);
  setStatus("waiting", "正在导出", "追加写入 human_replay_data.jsonl");
  try {
    game = await requestJson("/api/export-game", { gameId: game.gameId });
    renderGame();
    setStatus("ready", "导出完成", `已追加 ${game.exportedSamples || 0} 条样本到 ${game.exportPath || game.humanReplayPath}`);
  } catch (error) {
    setStatus("ended", "导出失败", error.message);
  } finally {
    setBusy(false);
    updateExportButton();
  }
}

function updateMoveCountSummary(currentGame) {
  if (!currentGame?.board) {
    moveCountLabel.textContent = "";
    return;
  }
  const count = currentGame.moveCount ?? countBoardMoves(currentGame.board);
  moveCountLabel.textContent = debugMode ? `Stones ${count}` : `Moves ${count}`;
}

function updateValueSummary(currentGame) {
  const hasNetworkValue = currentGame?.aiValue !== null && currentGame?.aiValue !== undefined;
  const hasMctsValue = currentGame?.aiMctsValue !== null && currentGame?.aiMctsValue !== undefined;
  if (!hasNetworkValue && !hasMctsValue) {
    valueSummary.textContent = "";
    return;
  }
  const prefix = showHint
    ? `${stoneName(currentGame.currentPlayer)} Hint Value`
    : debugMode
      ? `${stoneName(currentGame.currentPlayer)} Value`
      : "AI Value";
  const parts = [];
  if (hasNetworkValue) parts.push(`Net ${(currentGame.aiValue * 100).toFixed(1)}%`);
  if (hasMctsValue) parts.push(`MCTS ${(currentGame.aiMctsValue * 100).toFixed(1)}%`);
  valueSummary.textContent = `${prefix} ${parts.join(" | ")}`;
}

function countBoardMoves(board) {
  return board.reduce(
    (total, row) => total + row.reduce((rowTotal, value) => rowTotal + (value === 0 ? 0 : 1), 0),
    0,
  );
}

function updateSelectedMoveSummary(currentGame) {
  if (debugMode || showHint || !currentGame?.lastMove) {
    selectedMoveSummary.textContent = "";
    return;
  }
  const row = currentGame.lastMove.row;
  const col = currentGame.lastMove.col;
  const selectedPolicy =
    currentGame.aiSelectedPolicy ?? currentGame.aiPolicy?.[row]?.[col] ?? null;
  const selectedVisits =
    currentGame.aiSelectedVisits ?? currentGame.aiVisits?.[row]?.[col] ?? null;

  if (selectedPolicy === null && selectedVisits === null) {
    selectedMoveSummary.textContent = "";
    return;
  }

  const policyText = selectedPolicy === null ? "-" : `${(selectedPolicy * 100).toFixed(2)}%`;
  const visitsText = selectedVisits === null ? "-" : String(selectedVisits);
  selectedMoveSummary.textContent = `AI 选点 Policy ${policyText} | Visits ${visitsText}`;
}

function activeOverlayData(currentGame) {
  if (showHint && hintData) {
    return {
      policy: hintData.policy,
      visits: hintData.visits,
      visitTotal: hintData.visitTotal,
      aiValue: hintData.value,
      aiMctsValue: hintData.mctsValue,
      mctsBackend: hintData.mctsBackend,
      currentPlayer: hintData.currentPlayer,
    };
  }
  return {
    policy: currentGame?.aiPolicy || null,
    visits: currentGame?.aiVisits || null,
    visitTotal: currentGame?.aiVisitTotal || 0,
    aiValue: currentGame?.aiValue,
    aiMctsValue: currentGame?.aiMctsValue,
    mctsBackend: currentGame?.mctsBackend,
    currentPlayer: currentGame?.currentPlayer,
  };
}

function applyCheckpointDefaults() {
  const selected = checkpoints.find((checkpoint) => checkpoint.id === checkpointSelect.value);
  if (!selected) return;
  if (selected.mctsCandidateDistance === null || selected.mctsCandidateDistance === undefined) {
    candidateDistanceInput.value = "";
  } else {
    candidateDistanceInput.value = String(selected.mctsCandidateDistance);
  }
  evalBatchSizeInput.value = String(selected.mctsEvalBatchSize || 512);
  tacticalShortcutsInput.checked = selected.mctsTacticalShortcuts !== false;
  if (debugMode) resetDebugBoardFromCheckpoint(selected);
}

function setDebugMode(enabled) {
  debugMode = enabled;
  debugModeInput.checked = enabled;
  document.body.classList.toggle("debugMode", enabled);
  if (enabled) {
    const selected = checkpoints.find((checkpoint) => checkpoint.id === checkpointSelect.value);
    preDebugGame = game ? cloneGame(game) : null;
    clearHint();
    enterDebugMode(selected);
    showPolicy = true;
    showVisits = false;
    updatePolicyToggle();
    updateVisitsToggle();
    setStatus("ready", "Debug 摆盘模式", "点击棋盘摆子，检测网络输出");
  } else {
    if (preDebugGame) {
      game = preDebugGame;
      preDebugGame = null;
    }
    renderGameOrBoard();
    setStatus("ready", game ? "已退出 Debug" : "可以开始", game ? "可继续查看当前棋局或开始新局" : "选择 checkpoint 后开局");
  }
}

function enterDebugMode(selected) {
  if (game) {
    game = {
      ...game,
      board: game.board.map((boardRow) => [...boardRow]),
      currentPlayer: debugEvalPlayer,
      aiPlayer: debugEvalPlayer,
      status: "playing",
      winner: null,
      aiPolicy: null,
      aiValue: null,
      aiMctsValue: null,
      aiVisits: null,
      aiVisitTotal: 0,
      aiSelectedPolicy: null,
      aiSelectedVisits: null,
      winLine: null,
    };
    renderGame();
    return;
  }

  const boardHeight = selected?.boardHeight || 15;
  const boardWidth = selected?.boardWidth || 15;
  game = {
    gameId: "debug",
    checkpoint: checkpointSelect.value,
    board: Array.from({ length: boardHeight }, () => Array(boardWidth).fill(0)),
    boardHeight,
    boardWidth,
    humanPlayer: 1,
    aiPlayer: debugEvalPlayer,
    currentPlayer: debugEvalPlayer,
    status: "playing",
    winner: null,
    lastMove: null,
    aiPolicy: null,
    aiValue: null,
    aiMctsValue: null,
    aiVisits: null,
    aiVisitTotal: 0,
    aiSelectedPolicy: null,
    aiSelectedVisits: null,
    winLine: null,
  };
  lastMoveLabel.textContent = "Debug 摆盘";
  renderBoard(game);
}

function resetDebugBoardFromCheckpoint(selected) {
  if (!debugMode) return;
  game = null;
  enterDebugMode(selected);
}

function renderGameOrBoard() {
  if (game) {
    renderGame();
  } else {
    renderBoard(null);
    lastMoveLabel.textContent = "等待开局";
  }
}

function setPaintValue(value) {
  paintValue = value;
  paintBlack.classList.toggle("active", value === 1);
  paintWhite.classList.toggle("active", value === -1);
  paintEmpty.classList.toggle("active", value === 0);
}

function setDebugEvalPlayer(value) {
  debugEvalPlayer = value;
  evalBlack.classList.toggle("active", value === 1);
  evalWhite.classList.toggle("active", value === -1);
  if (debugMode && game) {
    game.currentPlayer = value;
    game.aiPlayer = value;
    game.aiPolicy = null;
    game.aiValue = null;
    game.aiMctsValue = null;
    game.aiVisits = null;
    game.aiVisitTotal = 0;
    game.aiSelectedPolicy = null;
    game.aiSelectedVisits = null;
    game.winLine = null;
    renderGame();
  }
}

function paintDebugCell(row, col) {
  if (!debugMode || !game || busy) return;
  game.board[row][col] = paintValue;
  game.lastMove = paintValue === 0 ? null : { row, col };
  game.aiPolicy = null;
  game.aiValue = null;
  game.aiMctsValue = null;
  game.aiVisits = null;
  game.aiVisitTotal = 0;
  game.aiSelectedPolicy = null;
  game.aiSelectedVisits = null;
  game.winLine = null;
  clearHint();
  renderGame();
}

async function runDebugPredict() {
  if (!debugMode || !game || !checkpointSelect.value) return;
  setBusy(true);
  setStatus("waiting", "正在检测网络输出", "前向计算 policy 和 value");
  try {
    const prediction = await requestJson("/api/debug-predict", {
      checkpoint: checkpointSelect.value,
      board: game.board,
      currentPlayer: debugEvalPlayer,
    });
      game = {
      ...game,
      board: prediction.board,
      boardHeight: prediction.boardHeight,
      boardWidth: prediction.boardWidth,
      currentPlayer: prediction.currentPlayer,
      aiPlayer: prediction.currentPlayer,
      aiPolicy: prediction.aiPolicy,
      aiValue: prediction.aiValue,
      aiMctsValue: prediction.aiMctsValue,
      winLine: null,
    };
    showPolicy = true;
    showVisits = false;
    clearHint();
    updatePolicyToggle();
    updateVisitsToggle();
    renderGame();
    setStatus("ready", "网络输出已更新", `${stoneName(debugEvalPlayer)}视角`);
  } catch (error) {
    setStatus("ended", "检测失败", error.message);
  } finally {
    setBusy(false);
    if (game) renderGame();
  }
}

async function toggleHint() {
  if (showHint) {
    clearHint();
    renderGameOrBoard();
    return;
  }
  if (!game || debugMode || game.status !== "playing" || game.currentPlayer !== game.humanPlayer || busy) return;

  setBusy(true);
  setStatus("waiting", "正在生成提示", "以当前玩家视角运行 MCTS");
  try {
    hintData = await requestJson("/api/hint", { gameId: game.gameId });
    showHint = true;
    showVisits = true;
    showPolicy = false;
    updateHintToggle();
    updatePolicyToggle();
    updateVisitsToggle();
    renderGame();
    setStatus("ready", "提示已生成", "可切换 Policy 或 Visits 查看提示结果");
  } catch (error) {
    setStatus("ended", "提示失败", error.message);
  } finally {
    setBusy(false);
    if (game) renderGame();
  }
}

function clearHint() {
  showHint = false;
  hintData = null;
  updateHintToggle();
}

function updateHintToggle() {
  const canHint = Boolean(
    game && !debugMode && game.status === "playing" && game.currentPlayer === game.humanPlayer,
  );
  hintToggle.disabled = busy || (!showHint && !canHint);
  hintToggle.classList.toggle("active", showHint);
  hintToggle.setAttribute("aria-pressed", String(showHint));
}

function updateExportButton() {
  const canExport = Boolean(game && !debugMode && game.status === "ended" && !game.exported);
  exportGameBtn.disabled = busy || !canExport;
  exportGameBtn.textContent = game?.exported ? "对局已导出" : "导出对局数据";
}

function backendStatusText(currentGame) {
  if (!currentGame?.mctsBackend) return "";
  const backend = currentGame.mctsBackend === "cpp" ? "C++" : "Python";
  const fallback = currentGame.mctsBackendNote ? "，C++ 不可用已回退" : "";
  return ` · ${backend} MCTS${fallback}`;
}

function formatProbability(value) {
  const percent = value * 100;
  if (percent >= 10) return `${percent.toFixed(0)}%`;
  return `${percent.toFixed(1)}%`;
}

function formatVisitCount(value) {
  if (value >= 1000) return `${Math.round(value / 100) / 10}k`;
  return String(value);
}

function overlayTitle(row, col, probability, visitCount) {
  const base = `${row + 1}, ${col + 1}`;
  if (showVisits) return `${base}: ${visitCount} visits`;
  if (showPolicy) return `${base}: ${(probability * 100).toFixed(2)}%`;
  return base;
}

function setBusy(nextBusy) {
  busy = nextBusy;
  settingsBtn.disabled = nextBusy;
  newGameBtn.disabled = nextBusy;
  debugPredictBtn.disabled = nextBusy;
  updateHintToggle();
  updateExportButton();
  for (const cell of boardEl.querySelectorAll(".cell")) {
    if (nextBusy) cell.disabled = true;
  }
}

function setStatus(kind, title, text) {
  statusDot.className = `dot ${kind === "ready" ? "" : kind}`;
  statusTitle.textContent = title;
  statusText.textContent = text;
}

function stoneName(value) {
  return value === 1 ? "黑棋" : "白棋";
}

async function requestJson(url, body) {
  const response = await fetch(url, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}
