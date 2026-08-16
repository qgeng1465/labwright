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
- **P1 LabMath-Bench [完成，待 commit]**：
  - **1a 五新 calc 全链路**：bioprinting/coculture/enzyme/bioinformatics(champ+plink)/
    solvent 六域，每域 = pydantic schema + Block（raw/derived/consistency/field_map/
    sanity_bands/canonical_units）+ derive_* + tools + sanity 带 + gold 全链路接入。
  - **1b TBA 指标**：`GoldExperiment.level`（默认 None 向后兼容）；`tba(records, τ)`
    按 (entry, key) 相对误差二值平均；report.py `derive()` 加 `tba` + `tba_by_level`
    （Wilson CI over key-pairs，旧结果 JSON 无 level 则不显示，向后兼容）。
  - **1c 数据集**：`make_labmath_bench.py` 生成 **510 条**（L1=170/L2=170/L3=170，
    hard 158/medium 207/easy 145，1590 可打分目标，seed=20260817 确定性）。
    `tag_existing_levels.py` 打标既有 7 个 gold 集 100 条 → 合并
    **610 条**（L1=213/L2=223/L3=174）。每条含 {id, goal, expected(裸 derived 键),
    source, level, scenario="complete-info", difficulty}，期望值由同一计算器经
    `submit_design` 算出，self-consistent。
  - **关键修正**：溶剂采样原窗口 [0.05,4.0]h 大片落在 d² 定律全蒸干区
    （1 µL 滴 37 °C/30%RH/edge 1.5 半小时即干）→ 43/56 条 residual=0.0，
    relative-error 除零毒化 TBA。修正 = `_sample_solvent` 拒绝采样至
    residual ≥ 15% V0（用实际 edge_well_factor），重生成后 0 退化目标。
  - **1d 测试**：`eval/test_labmath_bench.py` 5 测（数据集形状/键域/路由/确定性
    逐字节复现/合并文件）→ **pytest 552 绿**；audit_claims 新加 J 节 10 断言
    → **142 通过**。**commit 待做**（作者仅 qgeng1465）。
- **P2 Code Interpreter + 全消融 [待]**。
- **P3 边界对抗 + 主动提问 [待]**。
- **P4 工程规范 [待]**。
- **P5 终审 [待]**。
