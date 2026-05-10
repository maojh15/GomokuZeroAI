# GomokuZeroAI

GomokuZeroAI 是一个 AlphaZero 风格的五子棋训练与对战实验项目。它用策略-价值网络评估局面，用 MCTS 搜索选择落子，并通过自我对弈不断生成训练数据。

项目当前包含三条主要工作流：

- 训练模型：自我对弈、训练、保存 checkpoint、与上一代模型评估。
- 人机对战：本地 Web 页面加载 checkpoint，与 AI 下棋并查看 policy、visits、value 等调试信息。
- 模型对比：让两个 checkpoint 进行平衡先后手对局，输出胜负、得分率和步数统计。

## 快速开始

下面的步骤会下载已经训练好的权重，并启动本地人机对局页面。权重托管在 Hugging Face：

```text
https://huggingface.co/maojh15/GomokuZeroAI
```

默认下载的权重文件是 `iter_0150_15x15.pt`：

```text
https://huggingface.co/maojh15/GomokuZeroAI/blob/main/iter_0150_15x15.pt
```

1. 克隆项目并进入目录：

```bash
git clone https://github.com/maojh15/GomokuZeroAI.git
cd GomokuZeroAI
```

2. 安装依赖：

```bash
pip install numpy torch pyyaml huggingface_hub
```

3. 下载权重到本地 checkpoint 目录：

```bash
hf download maojh15/GomokuZeroAI iter_0150_15x15.pt --local-dir result_15x15/checkpoints
```

下载后文件应位于：

```text
result_15x15/checkpoints/iter_0150_15x15.pt
```

4. 启动人机对局服务：

```bash
python play_human.py --host 127.0.0.1 --port 8765
```

5. 打开浏览器：

```text
http://127.0.0.1:8765
```

页面会自动扫描仓库内的 `.pt` 权重文件。在 Checkpoint 下拉框中选择刚下载的模型，点击“开始新局”即可对弈。

人机对战默认优先使用 C++ MCTS 后端，并用 `eval batch size` 控制叶节点批量网络推理大小；如果 C++ 扩展不可用，会自动回退到 Python MCTS。想要更快的 AI 响应，可以把页面左侧的 `MCTS playouts` 调低；如果想要更强的搜索，可以调高，但每步会更慢。

## 功能概览

- 15x15 五子棋规则，棋盘大小可在配置中调整。
- PyTorch 策略-价值网络，输入为当前行棋方和对手的双通道棋盘。
- PUCT MCTS，支持纯 Python 后端和 C++ Torch Extension 后端。
- 自我对弈数据生成、replay buffer、旋转/翻转数据增强。
- checkpoint 自动保存和恢复训练。
- 评估时平衡先后手，并记录平均步数和最大步数。
- 本地 Web 人机对战界面，支持摆盘调试、Hint、Policy/Visits 叠加显示。

## 目录结构

```text
.
├── train.py                      # 训练入口，转发到 gomoku_zero.train
├── play_human.py                 # 本地 Web 人机对战服务
├── setup.py                      # 构建 C++ MCTS 扩展
├── train_config.yaml             # 默认训练配置
├── run_train.bat                 # Windows 训练脚本示例
├── gomoku_zero/
│   ├── config.py                 # TrainConfig 与 YAML 加载
│   ├── gomoku_rules.py           # 棋盘规则、合法落子、胜负判断
│   ├── policy_value_model.py     # 策略-价值网络
│   ├── mcts.py                   # Python MCTS
│   ├── cpp_mcts.py               # C++ 后端封装
│   ├── self_play.py              # 自我对弈
│   ├── replay_buffer.py          # 训练样本缓存和数据增强
│   ├── trainer.py                # 单轮训练
│   ├── checkpoint.py             # checkpoint 保存/加载
│   └── evaluate.py               # 模型对模型评估
├── gomoku_zero/cpp/
│   └── mcts_extension.cpp        # C++ MCTS 扩展源码
├── web_human/
│   ├── index.html                # Web 对战页面
│   ├── app.js                    # 前端交互逻辑
│   └── styles.css                # 页面样式
└── tests/
    ├── test_training_pipeline.py # 训练流程 smoke test
    ├── compare_ai.py             # checkpoint 对比工具
    └── plot_loss.py              # 日志/损失辅助脚本
```

## 环境准备

建议使用独立 Python 环境，并安装 PyTorch、NumPy。读取 YAML 配置时优先使用 `pyyaml`，如果没有安装，项目也内置了一个简单 YAML 解析器，可以处理当前配置文件这种标量格式。

```bash
pip install numpy torch pyyaml
```

如果使用 `mcts_backend: cpp`，需要先编译 C++ 扩展：

```bash
python setup.py build_ext --inplace
```

Windows 上构建 PyTorch C++ Extension 通常需要可用的 MSVC 编译环境。若暂时不想处理 C++ 编译，可以在配置里改成：

```yaml
mcts_backend: python
```

Python 后端更容易运行，但速度会慢很多。

## 运行测试

```bash
python -m unittest -v
```

如果默认配置或测试会走 C++ 后端，请先执行：

```bash
python setup.py build_ext --inplace
```

## 训练模型

使用默认配置训练：

```bash
python train.py --config train_config.yaml
```

训练流程每一轮会执行：

1. 用当前模型和 MCTS 生成自我对弈数据。
2. 将样本加入 replay buffer。
3. 从 buffer 采样训练策略-价值网络。
4. 保存当前 iteration 的 checkpoint。
5. 与上一代 checkpoint 进行评估。

checkpoint 默认写入配置项 `checkpoint_dir`。当前 `train_config.yaml` 中为：

```yaml
checkpoint_dir: result_15x15/checkpoints
```

如果目录中已经有同棋盘尺寸的 `iter_XXXX_15x15.pt`，训练会从最大 iteration 继续。例如已有 `iter_0004_15x15.pt` 且 `num_iterations: 10`，下次会从第 5 轮开始。

Windows 上也可以参考：

```bat
run_train.bat
```

这个脚本会激活指定 conda 环境，并把带时间戳的训练日志写到 `result_15x15/log.txt`。使用前请根据自己机器修改脚本里的环境名、配置路径和日志路径。

## 快速 Smoke Test 配置

如果只是验证环境和流程能跑通，可以临时把配置改小：

```yaml
num_iterations: 1
self_play_games_per_iteration: 1
self_play_workers: 1
mcts_backend: python
mcts_playouts: 2
epochs: 1
eval_games: 2
eval_workers: 1
channels: 16
```

这个配置不会产生有意义的棋力，只用于确认自我对弈、训练、保存和评估链路正常。

## 本地 Web 人机对战

启动服务：

```bash
python play_human.py --host 127.0.0.1 --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

页面会扫描仓库内的 `.pt` checkpoint，并在下拉框中列出可用模型。界面支持：

- 选择 checkpoint。
- 选择人类执黑或执白。
- 设置 MCTS playouts、eval batch size、`c_puct`、candidate distance、tactical shortcuts。
- 开启 `Policy` 显示网络策略概率。
- 开启 `Visits` 显示 MCTS 访问次数。
- 开启 `Hint` 查看当前局面推荐。
- Debug 摆盘模式：手动摆子后检测网络输出。
- 显示当前步数、网络 value、MCTS root value、AI 选点 policy/visits。
- 显示当前实际 MCTS 后端；默认使用 C++，扩展不可用时自动回退 Python。
- 对局结束后点击“导出对局数据”，把当前对局样本追加到 `human_replay_data.jsonl`。
- 可在设置里修改导出文件路径；路径必须位于仓库目录内，并使用 `.jsonl` 或 `.json` 后缀。

导出的每条样本只保存训练必需信息，并采用紧凑 JSONL 格式：

```json
{"p":-1,"s":[[112,1],[113,-1]],"pi":[[97,105],[112,936]],"z":0.0}
```

字段含义如下：

- `p`：该样本的当前玩家，也就是 AI 落子方。
- `s`：AI 落子前棋盘的稀疏棋子列表，元素为 `[move_index, stone_value]`。
- `pi`：AI 当步 MCTS visits 的稀疏列表，元素为 `[move_index, visit_count]`，保存的是访问次数，不是概率字符串。
- `z`：最终胜负得分，从 `p` 视角计算，取值为 `[0, 1]`。

`move_index = row * board_width + col`。训练加载时会根据配置中的棋盘尺寸还原 board，再生成 `state`，并把 `pi` 的 visit count 归一化成 policy。这个格式不兼容旧版 `human_replay_data.jsonl`；如果本地已有旧格式文件，请先删除或改名，否则训练加载时会报格式错误。

为避免把人类落子的 one-hot 选择当成搜索策略监督，导出时只保存 AI 落子前的局面和 AI 的 MCTS visits。后续运行训练时，程序会自动读取 `human_replay_data.jsonl`，放入单独的 human replay buffer。训练遍历数据时会把 human replay buffer 和自我对弈 replay buffer 看作一个逻辑上的合并数据集，但不会把 human 数据直接加入自我对弈 buffer，因此 human 数据不会被后续 self-play 样本挤掉。

## 对比两个 Checkpoint

使用 `tests/compare_ai.py` 可以让两个模型进行模型对模型比赛：

```bash
python tests/compare_ai.py path/to/checkpoint_a.pt path/to/checkpoint_b.pt --games 100 --playouts 2000 --workers 1 --seed 1243 --temp-threshold 12
```

输出包括：

- A/B 胜局数和胜率。
- 平局数。
- 平均步数和最大步数。
- A/B score rate，其中 score = win + 0.5 * draw。

两个 checkpoint 必须有兼容的棋盘大小、玩家编码、输入通道数和网络宽度。当前脚本内部使用 C++ 后端进行评估，所以运行前请先构建扩展：

```bash
python setup.py build_ext --inplace
```

## 关键配置说明

`train_config.yaml` 中常用字段如下：

- `board_height`, `board_width`：棋盘尺寸。
- `player_values`：棋子编码，默认 `[1, -1]`。
- `in_channels`：输入通道数，应与 `GomokuRules.encode_state` 输出一致。
- `channels`：策略-价值网络主干宽度。
- `device`：训练设备；设为 `null` 时自动选择 cuda/cpu。
- `seed`：Python、NumPy、PyTorch 随机种子。
- `checkpoint_dir`：checkpoint 保存目录。
- `num_iterations`：训练总轮数。
- `self_play_games_per_iteration`：每轮自我对弈局数。
- `self_play_workers`：自我对弈 worker 数。
- `mcts_backend`：`cpp` 或 `python`。
- `mcts_playouts`：每步 MCTS 模拟次数。
- `mcts_eval_batch_size`：C++ 后端批量评估叶子节点的 batch size。
- `mcts_candidate_distance`：只扩展已有棋子附近的空点；`null` 表示所有合法空点。
- `mcts_tactical_shortcuts`：是否对立即胜利和一手防守启用快捷判断。
- `c_puct`：PUCT 探索常数。
- `self_play_temp`：自我对弈开局采样温度。
- `self_play_temp_threshold`：前多少手使用 `self_play_temp`。
- `eval_explore_temp`：评估开局采样温度。
- `eval_temp_threshold`：评估前多少手使用 `eval_explore_temp`。
- `eval_temp`：评估后续落子的温度，通常接近贪心。
- `replay_buffer_size`：replay buffer 最大样本数。
- `augment_symmetry`：是否使用棋盘旋转/翻转增强。
- `human_replay_path`：人机对局导出的紧凑 JSONL 数据文件；设为空字符串可关闭加载。当前只支持 `p/s/pi/z` 新格式，不兼容旧版完整 `state/policy` 格式。
- `human_replay_buffer_size`：human replay buffer 最大样本数；会同样受 `augment_symmetry` 扩充。
- `batch_size`, `epochs`, `learning_rate`, `weight_decay`：训练超参数。
- `eval_games`：每轮训练后与上一代模型评估的局数。
- `eval_workers`：评估 worker 数。
- `log_interval`：每多少个优化 step 打印一次 loss；设为 `0` 关闭 step 日志。

## 数据表示和值函数

棋盘状态编码为两个通道：

- channel 0：当前行棋方的棋子。
- channel 1：对手棋子。

训练样本的 value target 始终从该样本的当前行棋方视角记录：

- 胜：`1.0`
- 平：`0.5`
- 负：`0.0`

模型 value head 使用 `Sigmoid()` 输出 `[0, 1]` 胜率估计。MCTS 内部使用 `[-1, 1]` 值域，因此会在进入搜索时进行转换：

```python
leaf_value = value * 2.0 - 1.0
```

Web 页面里显示的 `Net` value 来自神经网络直接预测，`MCTS` value 来自根节点平均价值并转换回 `[0, 1]` 胜率。

## 常见问题

### 找不到 C++ 扩展

如果看到类似 `gomoku_zero._mcts_cpp` 导入失败，先构建扩展：

```bash
python setup.py build_ext --inplace
```

或者把配置里的 `mcts_backend` 改成 `python`。

### Web 页面没有 checkpoint

`play_human.py` 会扫描仓库目录下的 `.pt` 文件。请先训练生成 checkpoint，或确认已有 checkpoint 位于仓库目录内，例如：

```text
result_15x15/checkpoints/iter_0001_15x15.pt
```

### CUDA 不可用

把配置里的：

```yaml
device: cuda
```

改成：

```yaml
device: null
```

或在运行脚本时指定 CPU 相关参数。`play_human.py` 的 `--device` 也可以传入 `cpu`。

### 训练很慢

优先检查：

- 是否已经构建并使用 `mcts_backend: cpp`。
- `mcts_playouts` 是否过大。
- `self_play_games_per_iteration` 和 `self_play_workers` 是否适合当前机器。
- GPU 是否被 PyTorch 正确识别。

更小的 `channels`、更少的 playouts 和更少的自我对弈局数可以让实验更快，但棋力也会下降。
