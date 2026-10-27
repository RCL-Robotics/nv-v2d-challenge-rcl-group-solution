# V2D Track 2 · HOI 估计工作流（hanzhou）

这个目录包含单目视频人-物交互（HOI）重建的完整工作流：代码、agent 任务说明，以及所有 agent 运行的对话记录。数据、权重、容器镜像和大文件都放在 NAS 上，不在这里，路径为 `/mnt/nas/Group/hanzhou/v2d/`。

## 流程

| 阶段 | 目录 | 内容 |
|---|---|---|
| 1. 基础 pipeline | `hoi_extraction/` | 基础 pipeline：<br>• 用 CARI4D / FoundationPose 逐帧估计物体位姿；<br>• 估计焦距、地面和相机；<br>• 各种后处理（对称翻转、时序平滑、地面约束等）。<br>其他内容：<br>• 打包 s1 发布：`scripts/build_release.py`；<br>• 评测脚本：`scripts/eval_object_pose.py`、`eval_official.py`；<br>• 容器定义：`singularity/`；<br>• 集群脚本：`slurm/`。 |
| 2. Agent in the loop | `agent/` | 每段视频交给一个隔离的 AI agent 修正，说明见 `agent/README.md`。<br>• 沙箱启动：`run_claude.sh`、`run_codex.sh`；<br>• 网络白名单代理：`allowlist_proxy.py`；<br>• 导出 agent 输入：`export_inputs.py`；<br>• 任务说明：各版本 `AGENT_TASK*.md`；<br>• 评分与审计：`score_runs.py`、`audit_run.py`；<br>• 打包 Opus v2 发布：`build_release_opus.py`；<br>• 收集记录：`collect_records.py`。 |
| 3. Tier 2 / Tier 3 精修 | `tier2_refine/` | • Tier 2 融合：`fuse_v3.py`，最好的变体 `v3_cf1_sym2`，AUC 0.781。<br>• Tier 3 第一帧锚定：`t3_frame0.py`，用第一帧物体位姿修对称翻转、相机旋转和常数平移。最好的变体 `opus 3 2 shift`，AUC 0.552。 |
| 4. 支撑面重建 | `support/` | 物体放在桌子或椅子上的 14 集，从视频中重建支撑家具，流程是背景中值、GroundingDINO、SAM2、贴地剪影拟合。<br>依次运行：<br>background → anchor → detect → segment → fit → contact → package，<br>最后用 `build_release_combined.py` 打包合并发布。 |
| 诊断 | `diagnostics/` | 只用于诊断的脚本。用到 GT 的只在这里，例如理想外参上限 `oracle_f0.py`、GT 相机拟合与渲染 `gt_camera_fit.py` / `gtcam_vis.py`。这些脚本不参与任何生成结果的流程。 |
| Agent 记录 | `agent_records/` | 全部 112 次 agent 运行，见 `agent_records/INDEX.md`。每次运行包括：任务说明、可读对话记录 `transcript.md`、原始事件流 `events.jsonl.gz`、运行清单、报告和求解脚本。 |

## 结果

指标为官方 ADD-AUC（1–9 cm），取 30 集平均，GT 只用于评测。

| 结果 | AUC |
|---|---|
| Tier 3 s1（基础 pipeline） | 0.210 |
| Tier 3 Opus v2（agent 修正） | 0.217 |
| Tier 3 Opus v2 + 第一帧锚定 | **0.552** |
| Tier 3 第一帧锚定 + 理想外参（诊断上限） | 0.612 |
| Tier 2 融合 `v3_cf1_sym2` | 0.781 |

下游发布包：`/root/autodl-tmp/han.zhou/tier3/release/tier3_frame0_supports_20261024/`，包含轨迹和支撑面。

## 规则
- 生成结果的流程和 agent 都不接触 GT。GT 只在 `diagnostics/` 和评测中使用。
- Tier 3 可以用的输入：单目视频、物体网格、地面、第一帧人体、第一帧物体位姿。
- agent 运行在 apptainer 沙箱里，网络走白名单代理。凭据只通过环境变量传入，不写入记录。打包时排除了 `.codex_home`，并扫描过常见的密钥格式。
