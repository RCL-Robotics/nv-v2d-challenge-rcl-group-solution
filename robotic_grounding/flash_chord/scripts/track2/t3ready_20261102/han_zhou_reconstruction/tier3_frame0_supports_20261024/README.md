# Tier 3 物体轨迹（第一帧锚定版）+ 支撑物重建 · 2026-10-24

目录格式与 `tier3_opus_20261021` / `tier3_s1_20261020` 完全相同，可直接作为 `soma_to_g1.py` 的 data_folder。

## 改了什么
- `reconstructed_data/<sequence_id>/poses.npy`、`object_pose_xyz_wxyz.npy`：换成目前最好的 Tier 3 物体轨迹，即 Opus v2 + 第一帧物体位姿修正（变体 `opus_f3_c2_shift`）。修正内容依次是：对称翻转、相机旋转误差、物体朝向锚定，以及常数平移，平移量取第一帧物体位姿和静止段均值之差。只用了规则允许的第一帧物体位姿，没有使用其他 GT。
- 新增 `supports/<sequence_id>/`：14 个放置类 episode（ep01 02 03 06 07 09 11 13 16 17 20 25 27 28）的支撑家具重建，包括网格、世界位姿、承载面碰撞盒和拟合检查图。字段说明见 `supports/README_supports.md`，其中 `object_poses_world.npz` 一项在本包里不再单独提供，就是上面的 `poses.npy`。
- `episodes.json` 和各集的 `episode.json`：`object_variant` 改为 `opus_v2_frame0_f3_c2_shift`；有支撑物的集多了一个 `support` 字段，指向对应的目录。
- 人体（`soma_params.npz`）、`camera.json`、`ground_plane.json`、`reconstructed_mesh/`、`frame_references/` 都没有改动。`agent_run/` 是 Opus v2 各集 agent 的报告，原样保留。

## 质量
官方 ADD-AUC（1–9 cm），30 集平均，只用于评测：

| s1 | opus_v2 | opus_v2 + 第一帧修正（本包） |
|---|---|---|
| 0.210 | 0.217 | **0.552** |

## 支撑物的仿真用法
1. 加载 `support_mesh.glb`，用 `support.json` 里的 `T_world_mesh_contact` 作为固定刚体的位姿；也可以直接用 `top_box` 作为承载面的碰撞盒。
2. `T_world_mesh_contact` 已经把家具沿竖直方向平移了 −5 到 +2 cm，使本包的物体轨迹静止时正好落在承载面上。

## 已知问题
- ep01：吧凳拟合精度较低（IoU 0.61）。
- ep17：锥桶的轨迹插进桌面约 5 cm，平移后桌面会比实际低 5 cm。
- ep28：结束时刮刀离吧凳约 10 cm，而 GT 中是贴着的。
