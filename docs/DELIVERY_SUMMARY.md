# Labwright — 交付总结

**日期**：2026-08-16 ｜ **HEAD**：`4526eca` ｜ **提交作者**：`qgeng1465` ｜ **状态**：GREEN，交付就绪

> 本文档是 50 小时自主马拉松的收尾交付物。除本节外，文中**每一个数字**都可由
> committed `results/*.json` 用 `eval.report.derive()` 或仓库自带分析脚本重算得出；
> 无单次 run 当结论，无发明生理值/DOI。全部数字的诚实边界见 §3。

---

## 1. 交付了什么

Labwright 是一个**器官芯片湿实验设计的「硬门禁」AI 副驾**。核心不变量：

> **任何进入方案的派生数字，都必须能由计算器从模型自己报告的原始输入重算出来**，
> 否则被门禁拒绝。

这使它区别于"AI 生成方案"——它产出的是**已验证**的设计（数字从自身输入自洽可重导），
并明确不假装解决"选哪个生理靶"（那是模型知识的范畴，见 §3）。

**双前端，共享同一道门禁**：

- **主链路（agent loop）**：`deepseek-v4-flash` / `deepseek-v4-pro`（温度 0.2、thinking
  默认关）跑 ReAct 工具循环（`max_iterations=12`、`max_tool_calls_per_turn=8`）。
  46 个注册工具（microfluidics/cell/culture/spheroid/dosing/o2/barrier/pk/stats/
  physiology/published 等），15 个 `calc/` 计算模块（`barrier/breathing/cell/culture/
  dosing/gradient/microfluidics/o2/pk/pulsatile/pumpless/scaling/spheroid/stats/units`）。
  四层验证器：① 算术重算 → ② 单位别名+生理软硬带 → ③ 安全（`SafetyConfig` 机构可配）
  → ④ prose 门（数字—文字一致性）。硬门禁 `submit_design`（reject derived →
  `DesignInput` extra=forbid → 重试环 → 溯源/ELN 导出）。
- **快速通道（fast-path）**：本地 `Qwen2.5-1.5B` LoRA 微调抽取器（v6 = 生产 adapter，
  与 `results/extractor/lora` 同权重），一步抽取参数，同一套计算器/验证器/门禁，
  不依赖外部模型。基准数字为 `labwright_iter`（≤3 次 fix-and-resubmit）。

**基准**：96 条 source-pinned 金标题 × 6 集合 = 24 reading + 15 blind（8 cold + 5
prompt-backed + 2 scenario）+ 14 culture + 15 spheroid + 14 pk + 14 新域。
指标：`可用率`（自洽 **且** 恢复金标目标 ±5%）、`自洽率`（零编数条目占比）、
`编数率`（幻觉均值；未提交/静默按约定记 1.0）。±5% 容差、一致性可从自身输入重算、
"数字写了但算不出"的字段不计入。

## 2. 性能

### 2.1 主链路 agent loop（96 题，六集合）

| 集合 | n | flash 可用/自洽/编数 | pro 可用/自洽/编数 | 剩余失败类型 |
|---|---|---|---|---|
| reading | 24 | 88% / 88% / 0.125 | 100% / 100% / 0.000 | — |
| blind | 15 | 40% / 100% / 0.000 | 47% / 100% / 0.000 | wrong_target（选错靶，非编数） |
| culture | 14 | 86% / 93% / 0.071 | 64% / 86% / 0.043 | 静默 / wrong_target |
| spheroid | 15 | 87% / 93% / 0.011 | 87% / 93% / 0.067 | wrong_target |
| pk | 14 | 79% / 100% / 0.000 | 86% / 100% / 0.000 | wrong_target |
| 新域 | 14 | 93%（13/14）/ 93% / 0.071 | 79%（11/14）/ 79% / 0.214 | 静默（梯度域两模型都答不出） |

关键：除 reading flash（2/24 未自洽）外，**所有提交方案编数率 = 0.000**；新域上
0.071/0.214 的编数率全部来自静默行（scorer 约定），无一编造。盲测失败全是
`wrong_target`——自洽但生理错的目标，门禁不放编数、但也补不了模型不知道的生理。

### 2.2 裸跑基线（同一批题，LLM 直接写数字、无门禁）

| 集合 | flash 可用 / 编数 | pro 可用 / 编数 |
|---|---|---|
| reading（24 题同款） | 0% / 0.833 | 0% / 0.917 |
| blind（旧 12 题集） | 0% / 0.667 | 0% / 1.000 |

注意：blind 裸跑基线是扩题前的 12 题集，与 Labwright 的 15 题不同集，只能定性对比；
reading 同集可比——**0 可用、9 成数字编数 → 门禁后 88–100% 可用、编数归零**。

### 2.3 快速通道 v6（本地 1.5B 抽取器）

| 集合 | 可用率 | 编数率 | repair 变体 |
|---|---|---|---|
| reading | 96% | 0.000 | 96% / 0.000 |
| culture | 57% | 0.143 | 57% / 0.143 |
| spheroid | 73% | 0.133 | 80% / 0.067 |
| pk | 50% | 0.500 | 50% / 0.500 |
| blind | 27% | 0.000 | 27% / 0.000 |
| newdomains | 29%（4/14） | 0.512 | 36%（5/14）/ 0.440 |

1.5B 抽取器在纯抽取任务（reading 96%）上超过主链路（88%），但领域知识集明显更弱
（blind 27%、newdomains 29%、pk 50% 且编数 0.500）。它是"便宜、确定性、可离线"
的前端，能力上限低于大模型主链路。

### 2.4 稳定性（种子级，非单次 run）

- **reading 5 种子 × 120 试次**：Labwright flash 可用 **0.925 [0.864, 0.960]**、
  pro **0.958 [0.906, 0.982]**（自洽 0.967 / 1.000）；裸跑 flash 0.067 [0.034, 0.126]、
  pro 0.108 [0.064, 0.177]；soft-gate 0.125/0.158；self-verify 0.000。**CI 永不重叠**。
- **盲测 3 种子 × 45 试次**：flash 0.444 [0.309, 0.588]、pro 0.489 [0.350, 0.630]，
  自洽 1.000；裸跑 0.000。cold-only（8 条冷题）flash/pro 均 **3/8 = 38%**
  （95% Wilson CI 14–69%）。
- **thinking 消融（各两遍，完全复现）**：pro 盲测 47% → **67%**（10/15）、
  flash 40% → 47–53%。4 条硬核靶（肾小管 0.02 Pa、肺动脉 2.0 Pa、视网膜 5.4 Pa、
  淋巴 0.2 Pa）两模型×两遍全 miss = **领域知识边界，不是推理预算**。门禁每遍
  自洽 100%、编数 0.000。
- **迭代修复（`labwright_iter`）**：verifier 在四集 **41 个条目上触发、41/41 全部
  修复、0 耗尽预算**；可用率与首提完全相同（43/58 = 74% 两者）。迭代是正确性回路，
  不是领域知识回路。

### 2.5 跨后端（门禁的可迁移性）

同一 harness 换 Kimi 后端：**k3** blind 47%、culture 93%、pk 86%、spheroid 73%
（≈ DeepSeek pro，编数≈0）；**kimicode** blind/culture/pk 0%、spheroid 33%、编数高。
诚实双结果：**门禁的好处只在"能可靠跑工具循环"的后端上迁移**；kimicode 基本不调
工具、直接报数，门禁拦下编数但可用率崩塌——这同时证明门禁真实起效而非空转。

### 2.6 外部真实数据（SciRecipe 审计）

21,094 篇已发表协议 → 14,589 篇含数字 → 5,700 篇被审计 → 457 篇声称有导出数字 →
**104 篇可复核 → 30 篇自洽（29%）**。真实世界协议大多无法复核（5,596/5,700
无 input），能在门禁标准下可验证的不足 2%——既说明标准严格，也说明文献本身多不自洽。
数据集：`qgeng1465/scirecipe-audit`（21,094 条，Crossref DOI 溯源）。

## 3. 诚实的边界（性能不等于什么）

1. **门禁验算术与内部一致性，不是生物学正确性**。`wrong_target` 类失败就是模型
   生理知识不足——门禁拦不住"自洽但错"的靶。
2. 盲测 recall 只有 40–67%（开 thinking），4 条硬核生理靶两模型都答不出；
   **这是当前最真实的短板**。
3. 快速通道在 pk（编数 0.500）与 newdomains（29%）上明确弱，作为负面结果写入论文。
4. 新域训练数据用"手写句式模板、刻意镜像评测题措辞"生成（register 迁移设计），
   三条机核防泄漏理由：46 条 gold 配对零新域；合成值采样非逐字；clean400 零重叠。
   诚实读法：新域 4/14 测的是"从同款表述吸收 schema"，不是"未见表述还能泛化"。
5. `--schema-prompt` A/B（精确键名清单换自然语 prompt）在 v4/v6 均 0/14——**廉价
   修复被证伪**，新域 gap 是 data/register 非 prompt。
6. 快速通道 repair 变体仅换入 `scaling-kidney-chip` 一题（4/14→5/14）；spheroid
   repair 提升可用率 73%→80% 但主链路/其余集无益。
7. 真实世界协议自洽率仅 29%，与文献"对标"需谨慎（§2.6）。

## 4. 可复现性

- **每步一个 commit，全部 commit 作者仅 `qgeng1465`，无任何自动署名附加**。
  历史如 §6.2。
- 所有 `results/eval_*.json` 已 commit；每张论文图（`paper/fig_*.py`）只读
  `results/` 与 `eval/`，`_check_render.py` 程序化验证 8 图 0 overlap。
- `eval/audit_claims.py`：**78 项机器断言**（从 committed JSON 用 `derive()` 重算并
  与 README/eval-README 显示值核对），接入 pytest，漂移即 CI 红。
- `eval/supervised_split.py`：gold-pair 成员法复现 seen/novel 拆分。
- `eval/analyze_iter.py`：复现 41/41 迭代修复计数。
- 报告 docx（`paper/report_to_teacher.py` 生成，gitignored）：`check_docx_text.py`
  转储 + AST 逐字校验（0 缺失），数字逐条对 committed JSON。
- 全量测试 **516 passed**（含 `test_audit_claims.py`、门禁攻击测试
  `test_gate_security.py` 等）。

## 5. 学术诚信审计（投稿前完整性）

- 终局独立审计 agent 裁决 **GREEN**（16 PASS 类）；发现的 5 项（2 P1 措辞精确度 +
  3 P2 整理）已全部修复并入 `4526eca`。
- 诚实修正历史：盲测 12→15 题集、cold-only 双 3/8、新域 13/14+11/14、silence
  行 hall=1.0 约定、schema-prompt 0/14 阴性、register-provenance 披露进 README/
  eval-README/教师报告/manuscript 四处。
- 无发明生理值/DOI：每个金标 `expected` 归入三桶之一（source-pinned DOI / 显式
  design-target / prompt-backed 锚值），无空桶、无第四桶。

## 6. 交付物清单

### 6.1 对外
- **GitHub** `qgeng1465/labwright`（main = `4526eca`）：完整代码、测试、基准、结果、
  README EN/ZH、8 张论文图源、训练数据生成器。
- **HF Space** `qgeng1465/labwright`：静态展示页（index.html + 3 张图 + README），
  与本地 byte-identical。
- **HF Dataset** `qgeng1465/scirecipe-audit`：21,094 条真实协议审计集。
- **论文侧（gitignored，不推）**：`paper/Labwright_工作报告_给导师.docx`、
  `paper/manuscript.md`（AJHG 投稿格式，摘要已更新至当前真数）。

### 6.2 关键 commit 链（作者仅 qgeng1465）
`379a178`（v6 数据+synthetic 模板）→ `0868ce0`（v6 六集基准）→ `c21857b`（README
EN/ZH v6）→ `ef265ad`（教师报告 v6）→ `29edd95`（fast-path 重述+seen/novel 修正）
→ `c3ae02c`（教师报告新域+摘要现代化）→ `51108ee`（15 题 thinking 消融真跑）→
`4cda257`（消融补进教师报告）→ `d3f0bb0`（清理）→ `2639fad`（audit_claims 78 断言+
register 披露+schema-prompt 阴性）→ `d9370af`（HF Space 同步）→ `8fa2190`（论文侧
register 披露）→ `4526eca`（投稿前审计 5 项修复）。

## 7. 未来工作

- **能力型改进的唯一合规路径** = gated v7 实验（gold-goal holdout + register-transfer
  血统披露的措辞迁移重训）。审计先验判断价值低（v3 已试过同一杠杆 0/14→4/14 后
  v5/v6 平住），仅在明确要求下启动，且必须作 holdout 门控，不得朝评测集训练。
- 其余（README 实验员重构、更大模型等）经审计判定为 filler 或数据受限，不在交付
  范围内。

## 8. 快速上手

```bash
pip install -e .
labwright design "灌流肝窦芯片，宽 1000 µm、高 100 µm，流速 0.5 µL/min，求壁面剪切力"
labwright verify-protocol "读入已发表协议反验证数字"   # 无需 API key
```

Gradio web / Colab demo 见 README；`.env` 配 `DEEPSEEK_API_KEY`（OpenAI 兼容即可）。
设计结果含 `derived`（已验证）与 `provenance`（公式+输入+单位+代码版本）字段，可导出
ELN。
