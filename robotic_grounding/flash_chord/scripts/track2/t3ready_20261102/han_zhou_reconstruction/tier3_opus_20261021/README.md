# Tier 3 物体轨迹 · Claude Opus agent 修正版（opus_v2，30 集）

目录格式与 `tier3_s1_20261020` 完全相同，可直接作为 `soma_to_g1.py` 的 data_folder。

- `reconstructed_data/<sequence_id>/poses.npy`、`object_pose_xyz_wxyz.npy`：替换为 agent 修正后的物体轨迹（世界系、网格文件坐标，约定同 s1）。
- `soma_params.npz`、`camera.json`、`ground_plane.json`、`reconstructed_mesh/`、`episodes.json`、`frame_references/`：与 s1 完全相同，人体未改动。agent 改了物体位置（多数集相对 s1 平均 2–6 cm），手-物相对位置与 s1 略有差别。
- `agent_run/<sequence_id>/`：每集 agent 的 REPORT.md（诊断、分阶段、验证表、置信度）、solve.py、TOOLS.md。

方法：Claude Code（Opus 5.5）在隔离沙箱中以 s1 为起点逐集修正（时序分段、多候选全局选择、渲染比对、物理约束），默认保留 s1 的整体深度。未使用 GT。

质量（官方 ADD-AUC 1–9 cm，仅评测用，30 集平均）：
| | s1 | opus_v2 | Tier 2 |
|---|---|---|---|
| 原始 | 0.210 | 0.217 | 0.432 |
| 只修常数平移 | 0.371 | 0.479 | 0.452 |

轨迹形状整体明显优于 s1（30 集中 23 集更好），s1 中的跳变被去除（如 ep2、ep16 s1 单帧跳 0.9–1.4 m，opus 最大 6–14 cm）。

使用注意（这些集建议改用 s1）：
- 对称物体被转到另一对称位置（朝向错 180° 或绕轴转错）：ep1 vacuum、ep14 round_table、ep26 skateboard。
- 整体深度被改错（形状好但位置偏约 9–13 cm）：ep3 contigo_coffee_mug、ep16 spray_bottle 的部分阶段。
- s1 与 opus 都失败：ep12 black_platform、ep15 wet_floor_sign。
