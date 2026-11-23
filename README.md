# Football TFT — 主客队门控融合的足球预期进球预测

本项目将原 Home Credit TFT 二分类项目改造为足球比赛进球数预测。模型输入每场比赛双方各自最近 10 场历史记录，先分别编码主队与客队，再通过门控融合形成 10 个融合时间步，最后送入 Temporal Fusion Transformer（TFT），分别预测主队和客队的预期进球数（xG）。

默认使用泊松分布（Poisson）作为计数概率模型，并从预测分布导出 P10/P50/P90
分位数。也可在配置中选择原有的负二项分布（Negative Binomial, NB）。

## 1. 数据结构与样本定义

项目默认读取：

- `data/a_a_a_feature.csv`：每场比赛 20 行历史特征。
- `data/a_a_a_test.csv`：每场比赛一行的比赛信息和真实比分。

数据检查确认：8,764 场比赛；每场恰好 20 条历史记录；seq_id 1–10 为主队历史，11–20 为客队历史；两张表 match_id 一一匹配。预处理还会检查历史比赛时间必须早于目标比赛时间，并检查主客队 ID 的方向一致性。

源数据中每队历史记录按最近到最早排列，预处理会分别反转两组序列，使模型按最早到最近的顺序读取。

## 2. observed / known / static 分类

### Observed：主客队各自 10 场历史

每场比赛有主队历史 10 条和客队历史 10 条。`home_history.npy`、`away_history.npy` 的形状分别为
`(样本数, context_length, 31)`；31 是每个历史时间步的特征数，不是时间步数。默认
`context_length=10`。

数值特征（27）：`win_rate_last_5, days_since_last_match, half_goals, half_goals_oppo, goals, goals_oppo, red_cards, red_cards_oppo, yellow_cards, yellow_cards_oppo, corner_kicks, corner_kicks_oppo, venue_capacity, group_count, round_count, shot_on_target, shot_on_target_oppo, shot_off_target, shot_off_target_oppo, attack, attack_oppo, danger_attack, danger_attack_oppo, possession, possession_oppo, blocked_shot, blocked_shot_oppo`

类别特征（4）：`competition_type, is_neutral, is_double_week, venue_surface`。数值特征只用实际训练子集拟合缺失值和标准化参数；类别先映射为整数，再分别嵌入后送入主、客队 GRU。

### Known：历史与目标比赛均可获得的时变特征

数值特征（每队 9 个）：`rank, rank_oppo, group_num, round_num, round_progress, pressure, temperature, wind, humidity`

类别特征（每队 1 个）：`weather`

主、客队各自的历史 known 特征按相同的“第几近”时间步并排输入，保留两路特征；目标比赛 decoder 步使用结果表中的赛前值。目标比赛客队的 `rank`/`rank_oppo` 按客队视角交换，其他共享的比赛字段两侧相同。`history_position` 仍作为一个归一化数值 known 特征。因此 known 张量共 19 个连续通道（含 `history_position`）和两个类别通道 `home_weather, away_weather`，默认形状为 `(样本数, 11, 21)`。

### Static：目标比赛不随时间变化的赛前上下文

数值特征（3）：`venue_capacity, group_count, round_count`

类别特征（2）：`competition_type, venue_surface`

### 类别嵌入 cardinality

类别有效取值数按 `[competition_type, is_neutral, is_double_week, venue_surface, weather]` 为
`[3, 3, 2, 4, 14]`。预处理为未知/缺失类别保留索引 `0`，有效类别从 `1` 开始，因此嵌入表大小（cardinality）分别加一：

- Observed 的类别嵌入（GRU 前，顺序 `competition_type, is_neutral, is_double_week, venue_surface`）：`[4, 4, 3, 5]`
- Known 的 TFT 类别输入（顺序 `home_weather, away_weather`）：`[15, 15]`
- Static 的 TFT 类别输入（顺序 `competition_type, venue_surface`）：`[4, 5]`

类别编码采用字段定义中的有效值范围；未知值映射为 `0`。`is_double_week` 在当前新特征 CSV 中实际编码为 `0/1`，预处理按这两个取值映射。
新训练模型的未知类别嵌入以零向量初始化，避免使用未经训练的随机向量；旧检查点
加载时仍保留原来的嵌入权重。

### 明确排除的字段

排除 `match_id, seq_id, id, team_id, team_id_oppo, competition_id, competition_name, match_time, venue_id, stage_id, season_id`。这些字段不进入模型。结果表中的 `home_scores`、`away_scores` 仅作为监督标签。盘口/赔率字段（`eu_*`、`rq*`）以及 `live*` 字段也不使用，避免模型依赖盘口或实时信息。

## 3. 模型结构

1. 主队历史序列 → Home GRU Encoder。
2. 客队历史序列 → Away GRU Encoder。
3. 每个历史时间步计算门控权重 `g`，执行 `g * home + (1-g) * away`，并加入主客交互投影。
4. 10 步融合表示作为 TFT 的 observed 输入；两队的 known 特征按时间步并排拼接后输入 TFT；目标比赛赛前上下文作为 static 输入。
5. TFT 输出隐表示，再由分布头预测主队/客队各自的进球分布参数。

默认泊松模型分别预测主客队的正数均值 `lambda`，条件方差也等于 `lambda`；损失为两队
泊松负对数似然之和（包含 `log(y!)` 项），在比赛间取平均。输出的 `pred_*_xg` 是条件
期望，不是对单场实际比分的确定性预测。

P10、P50、P90 是**进球数**的离散分位数：它们是概率分布累积概率首次达到指定比例的
最小整数，因此泊松或负二项模型的分位数都会是整数。例如均值为 1.5 的泊松分布，
P10/P50/P90 分别为 0/1/3；小数形式的条件期望仍保存在 `pred_*_xg` 列中。
单场比分点预测使用各队的 P50，分别保存到 `pred_home_goals` 和 `pred_away_goals`；
即使配置的 `inference_quantiles` 未包含 0.5，也会输出这两列。训练损失和 xG 均值不变。

可选负二项模型预测 `mu` 和 `alpha`，满足 `E[Y] = mu`、
`Var[Y] = mu + alpha * mu^2`。不同分布对条件方差的假设不同，改为泊松不保证误差下降。
推理依据检查点记录的分布计算分位数；不含分布字段的旧检查点按负二项模型加载。

## 4. 数据划分

默认按目标比赛时间升序排序：

- 前 6,000 场：训练池。
- 训练池最后 15%：作为时间顺序验证集。
- 训练池前 85%：实际梯度训练集。
- 剩余 2,764 场：独立测试集。

因此，验证集仍属于用户指定的前 6,000 场训练数据范围；测试集完全位于后续时间段。
缺失值填充和归一化统计量只使用实际梯度训练集拟合，不使用验证集或测试集。类别映射根据字段定义的有效取值范围固定，其他值归入未知类别 `0`。
旧版预处理数据需要重新生成；旧检查点仍须配合生成该检查点时的数据使用。

## 5. 安装依赖

Python 3.10+：

```bash
python -m pip install numpy pandas scikit-learn scipy torch
```

如需 CUDA，请为实际运行项目的 Python 环境安装 CUDA 版 PyTorch，并检查
`torch.cuda.is_available()`。若用户目录的 CPU 版 PyTorch 遮蔽了环境中的 CUDA 版，
可使用该环境的 `python -s -m ...` 启动命令，或调整环境中的包安装。

## 6. 运行步骤

在项目根目录执行。

预处理、训练和推理都读取 `configs/football_tft.json`；三个命令均支持
`--config <配置文件>`。配置中的相对路径相对于项目根目录解析，命令行显式指定的路径
相对于当前工作目录解析。

主要配置项：

- 所有运行配置集中在 `configs/football_tft.json`。输入字段名、每队历史行数和
  seq_id 起点在 `history_schema`；Observed/known/static 字段、类别有效值、标签列和排除列
  在 `features`；未知类别编号、数值回退值和 history_position 范围在 `preprocessing`。
  训练、模型、设备和推理参数与这些数据配置共用同一个 JSON 文件。
- 数据：`feature_path`、`results_path`、`processed_dir`、`train_size`、
  `context_length`。源数据每队有 10 场历史，`context_length` 可设为 1–10，
  使用每队最近的相应场数。修改 `train_size`、`context_length` 或 `validation_fraction`
  后必须重新预处理。
- 模型：`distribution`（`poisson` 或 `negative_binomial`）、`hidden_dim`、
  `num_heads`、`lstm_layers`、`gru_layers`、`gru_dropout`、`dropout`。
  `gru_dropout` 非零时 `gru_layers` 必须大于 1。旧配置未指定 `distribution` 时
  仍按负二项分布训练；更改分布必须重新训练，并相应修改 `checkpoint_dir`、
  `inference_checkpoint` 和 `inference_output`，避免覆盖已有结果。
- 训练：`validation_fraction`、`epochs`、`batch_size`、`learning_rate`、
  `weight_decay`、`optimizer_betas`、`optimizer_eps`、`scheduler_factor`、
  `scheduler_patience`、`scheduler_threshold`、`scheduler_cooldown`、`scheduler_min_lr`、`patience`、
  `grad_clip`、`seed`、`num_workers`、`device` 和 `checkpoint_dir`。
  `patience` 是早停耐心值，`scheduler_patience` 是学习率调度器的耐心值。
- 推理：`inference_checkpoint`、`inference_output`、`inference_batch_size`、
  `inference_device` 和 `inference_quantiles`。分位数必须在 0 和 1 之间；
  例如 `0.025` 输出列名为 `home_p2_5` 和 `away_p2_5`。推理使用检查点保存的模型参数，
  因此修改模型配置需要重新训练；`inference_checkpoint` 应指向对应的检查点。
- `device` 和 `inference_device` 支持 `auto`、`cpu` 和 `cuda` 等 PyTorch 设备名；
  `auto` 优先选择可用的 CUDA。

默认使用 `football_v2` 数据，并将新泊松模型保存到 `football_tft_poisson`，
不会覆盖旧版负二项模型结果。

### 第一步：数据预处理

```bash
python -m src.data.prepare_football
```

也可以通过命令行覆盖数据文件、输出目录和训练池大小：

```bash
python -m src.data.prepare_football --feature data/a_a_a_feature.csv --results data/a_a_a_test.csv --train-size 6000
```

命令行覆盖 `--train-size` 或 `--output-dir` 时，后续训练使用的配置也必须与之匹配。

### 第二步：训练

```bash
python -m src.training.train
```

指定配置：

```bash
python -m src.training.train --config configs/football_tft.json
```

调参时可加 `--skip-test`，仅使用验证集选择模型；最终确定参数后再评估测试集。
建议每次使用单独的 `checkpoint_dir`；跳过测试不会更新该目录中已有的测试指标。
推理要求预处理数据的 schema 与检查点一致，以免混用不同的归一化参数或类别映射。

训练输出：

- `artifacts/football_tft_poisson/best_model.pt`
- `artifacts/football_tft_poisson/training_history.json`
- `artifacts/football_tft_poisson/test_metrics.json`

### 第三步：测试集推理与分位数导出

```bash
python -m src.inference.predict
```

输出 `artifacts/football_tft_poisson/test_predictions_with_scores.csv`，包括：

- `match_id`
- `actual_home_goals`, `actual_away_goals`
- `pred_home_xg`, `pred_away_xg`
- `pred_home_goals`, `pred_away_goals`（P50 比分点预测）
- `home_p10`, `home_p50`, `home_p90`
- `away_p10`, `away_p50`, `away_p90`

### 第四步：导出让球胜平负概率

完成上一步后，在项目根目录运行：

```bash
python tests/lambda_to_handicap.py
```

脚本按 `match_id` 从 `results_path` 读取 `rq`，结合 `inference_output` 中的
`pred_home_xg` 和 `pred_away_xg`，保存到配置的 `handicap_output`（默认
`artifacts/football_tft_poisson/handicap_predictions.csv`）。原始预测文件不会修改。
结果包含 `rq`、主胜 `home_prob`、平局 `draw_prob`、客胜 `away_prob` 和概率最大的
`predicted_result`（`HomeWin`、`Draw`、`AwayWin`；概率相同时按该顺序选择）。

计算假设主客进球相互独立且分别服从泊松分布，胜平负按**主队进球 + rq 与客队进球**
比较，主队让球为负、受让为正。非整数 `rq` 的平局概率为 0；这是直接比较比分的规则，
**不是亚洲盘口四分之一球的拆盘结算**，不表示赢半或输半的概率。来源数据中的 `rq`
必须符合这一符号约定，且需确认是预测时可获得的让球值。

脚本支持 `--config`、`--input`、`--handicap-data` 和 `--output` 指定其他文件；
配置中的相对路径相对于项目根目录，命令行路径相对于当前工作目录。输出文件已存在时
会被覆盖，但不允许覆盖输入文件。

## 7. 评估指标

分别计算主队和客队 MAE、RMSE、WAPE，并记录所选分布的测试集 NLL。训练日志同时保存训练集和验证集指标。

## 8. 目录结构

```text
configs/
  football_tft.json
data/
  a_a_a_feature.csv
  a_a_a_test.csv
  processed/football_v2/    # 自动生成的 numpy 数组与特征 schema
src/
  data/prepare_football.py
  models/                   # TFT 原有组件 + football_tft.py
  training/
    losses.py               # Poisson / Negative Binomial NLL
    metrics.py
    trainer.py
    train.py
  inference/predict.py
artifacts/football_tft_poisson/  # checkpoint、日志、测试预测
```

## 9. 重要说明

- 本项目预测的是主队和客队的进球数条件分布均值（作为 xG 估计），并非传统事件级射门质量模型计算的 xG。
- 主客队编码器参数独立；门控在每个历史时间步动态决定双方信息对融合表示的贡献。
- 当前实现将主客队边际进球分布分别建模，未显式建模主客队进球之间的联合相关性。
- 数据中的天气、场地等赛前字段只有在实际预测时可获得的前提下才应保留；若这些字段是赛后回填值，应从 static 中移除。
- 时间划分按开赛时间排序，没有比赛结束时间或发布延迟信息，不能保证训练边界前比赛的
  最终比分在边界后第一场比赛开赛时已经可用；部署前应基于实际数据发布时间设置间隔。
