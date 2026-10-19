# E23 验证语义修订记录（V30 / V31）

**日期**：2026-10-19 ｜ **性质**：验证脚本语义修订，0 GPU、0 重训、不改任何结果
**脚本**：`scripts/e23_verify.py` ｜ **检查总数**：**45**（2026-10-19 第二轮新增 V39–V45；V17/V24 为原地加强；**未放宽任何数值容差**）

---

## 1. 为什么要改

2026-10-19 的两次**已授权**版本控制操作改变了六个 legacy 路径的状态：

| 提交 | 操作 | 影响 |
|---|---|---|
| `8fbe09e`（E24 提交） | 在 `.gitignore` 中忽略 `monitor/events.jsonl` | 该文件从 `git status` 消失 |
| `9781e9d`（脚本补齐提交） | 纳入 `scripts/e11_p4_runner.py`；提交 `monitor/exp_monitor.py` 的既有修改 | V30 的 dirty 计数下降，V31 的"从未提交"被打破 |

原 V30 断言「六个 legacy 文件**全部**仍 dirty/untracked（6/6）」，原 V31 断言「`8957237..HEAD` 内**任何** legacy 文件都未被提交」。这两条都是**历史时点断言**，在上述授权提交后必然失效：

- V30 实际值从 6/6 → **5/6**（`8fbe09e` 之后）→ **3/6**（`9781e9d` 之后）；
- V31 实际泄漏集合从 none → `{monitor/exp_monitor.py, scripts/e11_p4_runner.py}`（`9781e9d` 之后）。

因此 2026-10-19 的 `paper_ready` 复跑结果为 **36/38**，失败项恰为 V30、V31，**与文档/措辞修改无关**（V01–V29、V32–V38 全部 PASS）。

---

## 2. 六个 legacy 路径的状态表（逐项，非"3+2"式的结论先行）

| # | 路径 | 历史基线状态 | 当前状态 | 首次纳入 VCS | 授权依据 |
|---|---|---|---|---|---|
| 1 | `monitor/exp_monitor.py` | tracked + **modified**（长期未提交） | **tracked_clean** | `73eb8be`（原始加入）；修改于 `9781e9d` 提交 | 用户 2026-10-19 明确批准第三组提交包含 `monitor/exp_monitor.py` |
| 2 | `results/e8/status.json` | tracked + **modified** | **tracked_dirty**（保留原修改） | `85642c6` | 用户明确要求保留、不纳入本次提交 |
| 3 | `monitor/events.jsonl` | untracked（出现在 `git status`） | **untracked_ignored** | 从未 | `.gitignore` 规则（`8fbe09e`），用户批准"排除运行时日志" |
| 4 | `scripts/_lockprobe.txt` | untracked（6 字节探测残留） | **untracked** | 从未 | 用户裁决保留、不删除、不入库 |
| 5 | `scripts/_probe2.py` | untracked（6 字节探测残留） | **untracked** | 从未 | 同上 |
| 6 | `scripts/e11_p4_runner.py` | **untracked**（但被 E19–E23 验证脚本引用） | **tracked_clean** | `9781e9d` | 用户批准第三组提交补齐该脚本 |

**数量关系**：6 = **3 项有意保持未提交**（#2 tracked_dirty、#4/#5 untracked）+ **2 项经授权入库**（#1、#6）+ **1 项被 gitignore**（#3）。
该分解不是硬编码结论，而是由上表逐项事实推导；`LEGACY_DISPOSITION` 即此表的机器可读形式。

---

## 3. 修订后各检查保护什么

| 检查 | 修订前 | 修订后 |
|---|---|---|
| **V29** | 六个文件 md5 与 Stage-0 基线一致（文件缺失会以异常中断） | **不变的内容校验**；文件缺失现在记为 **FAIL 行**（`OSError` 被捕获），不再让脚本崩溃 |
| **V30** | 「6/6 仍 dirty/untracked」＝历史状态冒充当前状态 | **当前处置台账**：每项须等于 `LEGACY_DISPOSITION`；出现第 7 条未跟踪路径、已入库文件被再次修改、或文件消失 → FAIL |
| **V31** | 「`8957237..HEAD` 内无 legacy 泄漏」＝区间过宽，把 2026-10-19 的基础设施提交也算进去 | **两段式**：(a) **历史**区间固定为 `8957237..5b24193`（E21→E23，已核实无 legacy 路径）；(b) **检查点之后**只允许 `9781e9d` 触碰 `LEGACY_AUTHORISED_COMMIT_PATHS` 中的两个路径，且其**提交内 blob 的 md5 必须等于 Stage-0 基线**（防止"提交时顺手改内容"） |

**未做的事**：没有删除失败检查、没有放宽容差、没有撤销 `9781e9d` 的文件跟踪、没有把 HEAD 区间继续当成 E22/E23 区间使用。

**Git 失败处理**：`git()` 辅助函数改为在**非零退出码**时抛 `RuntimeError`（成功判定用 exit code，不用 stdout 是否为空，避免把"空输出"当通过）；V29/V30/V31 各自捕获异常并写入 FAIL 行。已用无效 commit 验证该分支会抛错。

---

## 4. 反向测试（不只测"应该通过"）

| 场景 | 期望 | 实测 |
|---|---|---|
| 授权集合置空（模拟未授权提交触碰 legacy） | V31 FAIL | **V31 FAIL**（rc=1） |
| 台账改错（`status.json` 期望 untracked） | V30 FAIL | **V30 FAIL**（rc=1） |
| 基线 md5 加入不存在的路径 | V29 FAIL，不崩溃 | **V29 FAIL**（`FileNotFoundError` 被捕获） |
| 无篡改基线 | 38/38 | **38/38，rc=0** |

---

## 5. 记录路径与历史记录的关系

- **历史记录**：`results/e23/verification.csv`（38 项 / 0 FAIL）保持**原样**，它只证明 2026-10-16 那个时点的状态，**不作为当前通过记录**。
- **语义修订前**：`results/paper_ready/verification_reconciliation_20261019.csv`（38 项 / **2 FAIL: V30/V31**），保留作为失败证据。
- **语义修订后**：`results/paper_ready/verification_reconciliation_20261019_post_semantics.csv`（38 项 / **0 FAIL**）。
- **勘误轮之后（2026-10-19 数值/措辞修订）**：`results/paper_ready/verification_reconciliation_20261019_post_errata.csv`（38 项 / **0 FAIL**）。本轮把 Doll 跨 seed 标准差统一为 `ddof=1` 的 **0.004638**（原 0.0037868 来自把 mean 行计入的 4 值 std）、Macro Avg 的六个 std 单元格留空、修正 E18 锚点、并记录 E25 MP `test0` 的字节级勘误。检查总数**仍为 38**：V24 增补「ddof=1 std 传播到 L5 / `doll_seed_summary.csv` / `paper_main_table.csv`」三项断言，V17 增补「六个 std 单元格必须为空」，**未放宽任何容差**。
- 复跑命令（务必带 `--out`，否则会覆盖历史 `results/e23/verification.csv`）：
  `python scripts/e23_verify.py --out <独立路径>.csv`

---

## 6. 已知残留不一致（如实保留）

- 历史 `verification.csv` 的 V30/V31 描述文字仍是"6/6 dirty"与"never added"，与其记录的 38/38 自洽，但**不再适用于当前状态**；本文档即为差异说明，不修改历史 CSV。
- 若未来再次改动 legacy 路径的跟踪状态，必须同步更新 `LEGACY_DISPOSITION` / `LEGACY_AUTHORISED` 并在本文件追加一节，**不得**通过删检查或放宽容差来恢复绿色。

---

## 7. 2026-10-19 第二轮：V30/V31 实质加固、E22 授权勘误与新增检查

**触发**：上一轮的 V30/V31 已是两段式，但实现层存在可被规避的问题；同时 E22 需要追加统计口径勘误，而旧 V01/V02 断言「results/e22 工作区完全干净 / 最后触碰提交恒为 `3d09b02`」会与之冲突。本轮逐条修正，并用隔离仓库反向测试证明检查确实能失败。

### 7.1 V30 修复（范围：六个 legacy 路径）

| 旧行为 | 问题 | 现行为 |
|---|---|---|
| `any(p in ln for ln in status.splitlines())` | 子串匹配（例如 `scripts/_probe2.py` 可被别的路径命中），也可漏判 | 解析 `git status --porcelain -z`：NUL 分隔、`XY PATH` 为**单个字段**，仅 R/C 追加旧路径字段；按**精确路径**比对 |
| 缺失文件按 tracked/dirty 归类 | 删除会被误判为 dirty，甚至被当成 untracked | 先检查磁盘存在性 → 独立的 `missing` 状态，台账不符即 FAIL |
| `check-ignore` 只看 `rc == 0` | `rc = 128`（非仓库 / 路径错误）被当作"未忽略" | 0 = 忽略、1 = 未忽略、其他 → `GitFailure` |
| 只报告台账六项 | 措辞易被读成"全仓审计" | 检查名明确 **NOT a whole-repo audit**；额外未跟踪文件仅作 `extra_untracked` 报告 |

> 修复过程中确实发现并修正了一个真实解析 bug：早期实现对每个条目误加 2 而不是 1，会**跳过一半的 status 条目**，从而可能把 dirty 的 legacy 文件读成 clean。R11 即为该 bug 的回归测试。

### 7.2 V31 修复（逐提交，不是路径集合）

1. 端点用**完整 40 位 hash** 解析，并断言 subject（`E21: validate R1-ILA across M2AD views` / `E23: audit Doll failure case and consolidate paper evidence`），短 hash 或改名无法重定向区间。
2. 历史区间 `89572372..5b24193f` 逐提交 `diff-tree`，任何 legacy 变动即 FAIL。
3. 检查点之后逐提交核验：commit 身份、路径、变动类型、**该提交对象内的 blob md5**（`git show <commit>:<path>`），不使用 index 或 HEAD blob。
4. 仅 `9781e9d` 的两个授权路径可动；类型仅 **A/M**；其他提交即使触碰同样路径也 FAIL。
5. **改后还原**：每个 A/M 提交都记录 blob，`R06` 证明「先改坏再改回」仍 FAIL。
6. 删除（D）与重命名（R）直接 FAIL；合并提交按**第一父**核验，且合并对 legacy 的任何变动都 FAIL。
7. 所有 Git 子进程检查退出码；失败写入 FAIL 行（`R09` 证明非仓库路径抛 `GitFailure`，不会退化为"无泄漏"）。
8. 每次运行报告扫描范围（当前：**4 个历史区间提交 / 1 个触碰 legacy 的检查点后提交**）。**只统计触碰 legacy 的 post 提交**，否则记录本轮验证的提交自身会改变计数值，形成"记录提交→观察值变化→再提交"的无限循环。

### 7.3 E22 授权勘误与检查语义

- **历史事实**：`3d09b02d5b86...` 归档了 E22 全部产物；V01 仍校验该提交的全 hash 与 subject。
- **当前完整性**：V01 分别检查 (a) `3d09b02..HEAD` 中除两份指定文档外**没有任何** E22 路径被触碰；(b) 工作区中这些冻结路径没有未提交改动。
- **唯一豁免**：V02 要求 `README.md` / `statistical_summary.md` **当前字节 = 归档 blob 字节 + 固定附录字节**（附录 sha256 硬编码），并检查附录必须包含的语义短语；删除、改写、重命名、替换或第三份文档都不通过。
- **没有"E22 所有 Markdown 任意变化"白名单**；两份文档的历史正文按字节保留。

### 7.4 新增 / 加强的检查（38 → 45）

| ID | 内容 |
|---|---|
| V17 | （加强）Macro Avg 六个 std 单元格必须为空；同一判定函数对"填入 0"的合成行必须返回 False |
| V24 | （加强）Doll `ddof=1` std 由三个原始 cell 复算，并要求同时出现在 L5 / `doll_seed_summary.csv` / `paper_main_table.csv` |
| V39 | Doll 全部 12 个数值列的 std/min/max 从三条原始 seed cell 复算（不信任 CSV）；`analysis_a` 仅读 raw |
| V40 | 包 README 明确"空白 = 未定义，绝不是 0" |
| V41 | E18 锚点 = seed0 F0→Ours（Bird/Car/Cube）；三种子均值列独立命名；E24 S15 seed0 复现同 6 值；EVIDENCE_MAP 文本一致 |
| V42 | `p_d_auroc_gt0` 实现为 `(das > 0).mean()`；论文面向文档写 P(ΔAUROC>0)，不写 `p = 0.001` |
| V43 | E25：39 = 33+3+3；split 等权与子数据集等权各自口径；PCB 绝对 0.444722→0.463074 已报告；重训而非迁移措辞 |
| V44 | V30/V31 隔离反向测试（11 场景）全部按预期 FAIL/PASS |
| V45 | 生成器在隔离 OUT 目录逐字节复现 6 份文本产物（不重生成图 / PDF） |

### 7.5 反向测试记录（`scripts/verification_reverse_tests.py`）

| 场景 | 期望 | 实测 |
|---|---|---|
| R01 缺失文件 | V30 FAIL | FAIL（`state=missing`） |
| R02 台账不符 | V30 FAIL | FAIL（`tracked_dirty`） |
| R03 额外未跟踪 | V30 PASS（单独报告） | PASS |
| R04 授权 A/M + 基线字节 | V31 PASS | PASS |
| R05 未授权后续修改 | V31 FAIL | FAIL |
| R06 改后还原 | V31 FAIL | FAIL（记录该提交 blob） |
| R07 未授权删除 | V31 FAIL | FAIL（D） |
| R08 未授权重命名 | V31 FAIL | FAIL（R100） |
| R09 Git 命令失败 | FAIL，不静默 | `GitFailure` |
| R10 合并提交触碰 legacy | V31 FAIL | FAIL（`merge=True`） |
| R11 多状态条目中的 dirty legacy | 正确识别 | PASS（`tracked_dirty`） |

**记录**：`results/paper_ready/verification_reverse_tests_20261019.csv`；完整验证：`results/paper_ready/verification_reconciliation_20261019_e22_erratum.csv`（45 项 / 0 FAIL）。

---

## 8. 2026-10-19 第三轮：记录字节级可复现性的两处修复（V30/V31 语义不变）

**触发**：提交 `ae27a3b` 之后按既定流程复跑两个脚本：

- 反向测试记录有 **1 行不同**（R05 的临时 SHA）；
- 完整验证记录**当次**恰好逐字节一致，但逐行核查 V02 后发现其 `observed` 字典由 `set(E22_AUTHORISED_DOCS)` 迭代构造，键顺序随 `PYTHONHASHSEED` 变化。修复前连续两次运行已经实测出现两种顺序（v1 记录为 `statistical_summary` 在前，另一次为 `README` 在前）。因此那次"一致"**不是可复现，而是碰巧**。

即：两处都是"记录的渲染依赖非确定性输入"，与检查语义无关。

### 8.1 事实 A（R05：临时仓库 SHA）

```text
< R05 unauthorised later modification,FAIL,FAIL,True,unauthorised=['2883e90']
> R05 unauthorised later modification,FAIL,FAIL,True,unauthorised=['d7f4ad9']
```

- 11 个场景数量一致；`scenario / expected / observed / match` 四列**完全一致**；
- 唯一差异是 R05 `detail` 中渲染的 **7 位短 hash**，来自 `tempfile.TemporaryDirectory()` 内的临时仓库，其 SHA 依赖时间戳，**每次运行必然不同**。

### 8.2 事实 B（V02：set 迭代序）

- 源码：`doc_paths = set(E22_AUTHORISED_DOCS)`，随后 `{p: ... for p in doc_paths}` 直接进入 `observed` 字段；
- 同一个 CSV 的 `baseline_sha256` / `appendix_sha256` 两个子字典，其**键顺序**在不同进程间不同，但值、判定（`prefix_ok` / `hash_ok` / `phrases_ok`）完全相同；
- 因此 V02 的**结论**从未随机，随机的只是**记录文本**；但足以让"复跑逐字节相等"这一契约失效。

### 8.3 修改（2 处，均不改判定）

| 项 | 修改前 | 修改后 |
|---|---|---|
| R05 `detail` | `unauthorised=['2883e90']`（临时 SHA） | `unauthorised=1 flagged_injected_commit=True` |
| V02 迭代源 | `doc_paths = set(E22_AUTHORISED_DOCS)` | `doc_paths = sorted(...)`；另保留 `doc_set` 供 `hist_paths - doc_set` 等集合运算 |

- **断言未变**：R05 的 `mechanism` 仍是 `any(h["commit"] == d for h in res["unauthorised"])`，即"被判定的提交**恰好**是测试注入的那个提交"；该布尔值原本就已参与 `match` 计算，现在只是把它写进 `detail`。V02 的 `all(...)` 本来与顺序无关，现在仍是同一组断言。
- **未变**：场景数（11）、期望值（9 FAIL / 2 PASS）、`match` 判定函数、V30/V31 审计逻辑、45 项完整验证的任何一条检查与容差。
- **未变**：R06 的 `unauthorised=2`、R07/R08 的 `types=[...]`、R10 的 `post=[...]` 均为确定性字段，保持原样。

### 8.4 验证结果

| 检查 | 结果 |
|---|---|
| 反向测试连续两次运行 `cmp` | **逐字节一致**（BYTE-STABLE） |
| 完整验证连续两次运行 `cmp` | **逐字节一致**（BYTE-STABLE） |
| 与 v1 记录对比 | 反向测试仅 R05 `detail` 一列不同；完整验证仅 V02 一行（键顺序）不同 |
| `match` / `pass` 全为通过 | 是（11/11、45/45） |
| V31 扫描范围 | 仍为 `4 hist / 1 legacy-touching post`（本轮提交未触碰 legacy 路径） |

### 8.5 记录版本与契约

| 文件 | 状态 | 说明 |
|---|---|---|
| `verification_reverse_tests_20261019.csv` | **保留原样**（v1） | 首次记录；`detail` 含一次性临时 SHA，**不可逐字节复现** |
| `verification_reverse_tests_20261019_v2.csv` | **新增**（v2） | 当前契约记录；可逐字节复现 |
| `verification_reconciliation_20261019_e22_erratum.csv` | **保留原样**（v1） | 45 项 / 0 FAIL；V02 键顺序为修复前的一次偶然顺序 |
| `verification_reconciliation_20261019_e22_erratum_v2.csv` | **新增**（v2） | 45 项 / 0 FAIL；V02 键顺序为 `sorted()` 后的固定顺序，可逐字节复现 |

**契约边界**：

- 反向测试的可复现承诺适用于 `scenario / expected / observed / match` 四列（v1、v2 完全相同）；`detail` 列自 v2 起亦纳入该承诺。v1 的 `detail` 仅作历史诊断信息，**不作为契约字段**。
- 完整验证的可复现承诺以 v2 为准；v1 与 v2 的差异仅限 V02 `observed` 的键顺序，判定列逐行相同。
- 任何未来新增的 `detail` / `observed` 都不允许渲染临时仓库的 SHA、时间戳、绝对路径或**未排序的集合迭代序**。
