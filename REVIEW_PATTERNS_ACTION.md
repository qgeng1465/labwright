# Labwright Patterns 审稿意见落地马拉松 — 进度真相源

（用户指令 2026-08-17：把 Patterns 风格审稿意见的三大维度干实验 + 工程规范全部落地。
本文件 = 唯一进度真相源。每次唤醒/推进都读它、改它。）

## 硬约束（每次推进前默读，违反即回滚）

- **诚实协议**：任何写进 README / 图 / 报告 / 论文的数字，必须能从
  `results/*.json` 或 `eval/gold_*.json` 的 source-pinned 值/DOI 逐条追溯；
  生成条目必须 self-consistent（期望值由同一计算器算出）；禁止编造生理值/DOI/引用。
- **审计门**：每阶段结束必须过审计才 commit —— `pytest` 绿 + 数字逐条核对。
- **作者纪律**：所有 commit 仅 `qgeng1465 <qgeng1465@users.noreply.github.com>`，
  无 Co-Authored-By / Claude / Anthropic 字样。`*.docx`、`paper/manuscript.md` 在
  gitignore，永不推。
- **资源**：重任务前 `arbitrate.py status`；V100/长跑用 `arbitrate.py run --detach`。
- **顺序**：先 commit 结果 JSON + 跑通审计函数，再改文档（防 auto-mode 分类器拒）。
- **MARATHON_50H.md 禁重建**：那是用户的文件，本马拉松只用本文件。
- **中文一律全角标点**。

## 用户拍板（2026-08-17）

1. 模型线保持 DeepSeek flash/pro + Thoth-8B，论文诚实注明「以同等可复现模型替代 GPT-4/Claude 3.5 Sonnet」。
2. LabMath-Bench 生成并全跑 ~500 对。
3. 后台马拉松自主跑（cron 推进、逐阶段审计、本地 commit 不推）。

## 阶段（顺序推进）

- **P0 基线验证**：pytest 516 绿确认；GPU 空；API key 在。
- **P1 LabMath-Bench**：五新 calc（bioprinting/coculture/enzyme/bioinformatics/solvent）
  全链路（Block/schema/derive/tools/sanity/gold）+ GoldExperiment.level + TBA 指标
  + make_labmath_bench.py 生成 ~500 对 + 打标既有 gold + 测试审计 commit。
- **P2 Code Interpreter + 全消融**：code_interpreter 沙箱系统（Baseline B）+
  全消融跑（bare/code_interpreter/labwright 全量 500 × flash/pro；扩展消融与
  thoth 分层子集）+ 混淆矩阵 CER→0 + fig_tba/fig_ablation。
- **P3 边界对抗 + 主动提问**：request_info 工具（elicit 开关）+ gold_adversarial.json
  ~30 条（missing_parameter/physical_conflict/lethal_condition）+ fail-safe 指标 +
  sanity 补带 + 全跑 + fig_failsafe。
- **P4 工程规范**：fig_protocol_dag.py（DAG 信息流 + 物质守恒）+ supplementary 溯源
  日志 + requirements.txt + Dockerfile + docs/PLUGINS.md + scripts/reproduce_all.sh
  + README 文档同步 + audit 断言。
- **P5 终审**：全量 pytest + audit_claims + 本地 commit（不推）。

## 进度

- **P0 [完成]**：基线验证（pytest 516 绿、GPU 空、API key 在）。
- **P1 LabMath-Bench [完成]**（commit 3b0a26f）：
  - 五新 calc 全链路（bioprinting/coculture/enzyme/bioinformatics/solvent）；
    `GoldExperiment.level`（默认 None）+ TBA 指标（τ=0.05 为主，report.py 按 level
    Wilson CI）；生成 510 条 + 打标既有 → 合并 610 条（L1=213/L2=223/L3=174）；
    溶剂采样 reject 到 residual≥15% V0 防除零。pytest 552 绿、audit J 节 10 断言。
- **P2a Code Interpreter 沙箱 [完成]**（已并入 f69efc1）：
  - `eval/benchmark.py` 新系统 `code_interpreter`（Baseline B）：LLM 输出 Python 片段
    算 `RESULT = {...}`，沙箱子进程 `[sys.executable, -c, code]` 执行，timeout 30s +
    rlimit 限制，失败分类 `code_exec_error`（语法/运行错）区别于 silence/wrong_target。
    60 条 benchmark 测试绿。
- **P2b 全消融跑 [进行中]**：
  - flash-core（610×bare/code_interpreter/labwright，pid 2263496，02:13 起，~19s/条，
    预计 ~3h）写入 `results/eval_labmath_flash.json`（新代码带 plan+provenance+tool_trace）。
  - ext-flash（102×soft_gate/self_verify/tool_no_gate/labwright_iter，pid 2263555，
    ~40min 完成）写入 `results/eval_labmath_ext_flash.json`。
  - **pro-core 全量 610 待 flash-core 完成后起跑**；thoth-8b 分层子集（V100 claim）。
  - `_score_design` 现存全 plan dict + 复验 provenance + tool_trace。
  - 混淆矩阵图 `paper/fig_ablation.py` 已写，等 pro 结果。
- **P3a/P3b 边界对抗 + 主动提问 [完成]**（并入 f69efc1）：
  - `request_info` 工具 + `DesignAgent.elicit` 开关（默认 False 保兼容）；
    gold_adversarial.json 30 条（missing_parameter/physical_conflict/lethal_condition），
    验证器硬拦截 18/18。
- **P3c 对抗全跑 [flash 完成 / pro 进行中]**：
  - flash 30/30 完成并 commit（fail_safe labwright 0.933 / bare 0.833 /
    code_interpreter 0.733；elicitation 0.667；exception_catch 0.233；fabrication 0.067）。
  - pro 进行中（pid 2263907，~7min 完成）→ 完成后跑 fig_failsafe + pin pro 审计值。
- **P4a 工程规范 [完成]**（并入 f69efc1）：requirements.txt + Dockerfile（非 root）
  + docs/PLUGINS.md（第三方 calculator 扩展契约全链路）。
- **P4b DAG 图 + 溯源日志 + 复现脚本 [完成，已 commit f69efc1]**：
  - `paper/fig_protocol_dag.py`：3 面板（a 管道拓扑 / b 字段级 provenance DAG 74 节点
    85 边，derived→derived 边带流值 / c 守恒审计 cells+seed 恒等式真验）。修 phantom
    tick label 重叠（axis-off 轴清空 tick 文本）→ **_check_render 0 overlap 全 8 图**。
  - `eval/make_traceability_log.py` + 4 测：从结果 JSON 逐条重建溯源日志
    （plan+provenance+tool_trace → supplementary/traceability/{model}/...json + INDEX +
    README），诚实统计 no-plan/plan-without-prov。
  - `scripts/reproduce_all.sh`：FULL=1 全量复现 / 默认 5 条冒烟；gold 确定性已验证
    （MD5 逐字节）。
  - `audit_claims`：audit_adversarial 钉 flash 精确值（<1e-4）+ 新 audit_traceability
    机制/覆盖断言。当前 **174 通过 / 4 失败**（4 失败全是 pro 对抗未跑完的预期项）。
- **P4e 文档同步 [待]**：README/README.zh-CN/eval-README 加 LabMath-Bench 节 + TBA 表
  + 消融混淆矩阵 + fail-safe 对抗 + GPT-4/Claude 替代诚实注 + 插件/Docker/复现链接。
- **P5 终审 [待]**：全量 pytest + audit 绿 + 结果 JSON 全 commit + 文档同步 + 本地
  commit（不推）。

## 后台跑批状态（2026-08-17 03:20）

| 批 | 内容 | pid | 起跑 | 进度 | 预计 |
|---|---|---|---|---|---|
| ~~adv-pro~~ | 30×3 系统 | 已完成 | 02:14 | **30/30 ✓** | 已落地 7ea4c97 |
| ~~ext-flash~~ | 102×4 系统 | 已完成 | 02:13 | **102/102 ✓** | 已落地 a7b1e60 |
| flash-core | 610×3 系统 | 2263496 | 02:13 | 172/610 | ~05:30 |

**ext-flash 已落地（a7b1e60）**：102 条分层子集 × soft_gate/self_verify/tool_no_gate/
labwright_iter。**门控消融故事成立**：soft_gate usable 0.029 / self_verify 0.010（halluc
~0.80）在 L1-L3 新域无硬验证器门即崩；tool_no_gate usable 0.902 / halluc 0.000、
labwright_iter usable 0.912 / halluc 0.010 胜出 → 硬 verifier 门是数字可信的来源。

落地顺序：flash-core 完 → 起跑 pro-core 610 + commit flash 结果；最终 pytest/audit 绿 +
文档 + 本地 commit（不推）。

**2026-08-17 05:22 更新**：
- **flash-core 已落地（783ff9e）—— 主线核心数字（610 条全量）**：

| 系统 | usable | halluc | TBA(0.05) | 混淆矩阵 CER |
|---|---|---|---|---|
| bare-LLM | 0.051 | 0.765 | 0.406 | 536/610 计算错（87.9%） |
| code_interpreter | 0.180 | 0.602 | 0.664 | 484/610 计算错（79.3%） |
| **labwright** | **0.934** | **0.000** | **0.965** | **0/610（CER→0）** |

  按 level：labwright TBA L1 0.950[0.930,0.964] / L2 0.955[0.936,0.968] /
  L3 1.000[0.992,1.000]，全部 > bare（0.512/0.451/0.191）。labwright 40/610
  wrong_target = 诚实参数提取 miss，非计算错。另修：derive() 补 code_interpreter
  （7e5f5cc）+ audit_labmath_results 全量诚实门（e386cb7）。
- **pro-core 已起跑**：pid 2320909（05:18 起，610×3，~09:10 预计，05:49 时 80/610）。
