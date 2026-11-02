# V2D Track 2 · Tier 3 重建结果（版本 s1，2026-10-20）

本目录是我们自己从**单目视频**重建出的 30 集人体 + 物体轨迹，供下游做 SOMA → G1 retarget 与策略训练。
格式直接对齐官方 `robotic_grounding/scripts/retarget/soma_to_g1.py` 的输入。

- 物体：CARI4D（GroundingDINO → SAM2 → MoGe-2 → SAM-3D-Body → FoundationPose → CoCoNet）
  + 相机联合标定（30 集共用内参 f=820, cx=758.1, cy=560.5）+ 脚踩地面深度校正 + FoundationPose/CoCoNet 深度融合。
- 人体：SAM-3D-Body / CoCoNet 的 MHR 人体，用同一套相机放到世界系，再用官方 SOMA-X `PoseInversion` 转成 SOMA 参数。
- 用到的 GT：**只有第 0 帧参考人体**，用于把重建对齐到官方世界坐标系（单个刚体变换）以及估计相机主点。
  没有使用第 0 帧之后的任何干净轨迹，也没有用 Tier 1/Tier 2 轨迹做训练或调参。
- 物体质量（官方 ADD-AUC，1–9 cm）：30 集平均 **0.210**（Tier 2 为 0.433），7/30 集超过 Tier 2，MPPE 11.0 cm。
  各集差异很大，见文末表格。

## 目录结构

```
tier3_s1_20261020/
├── README.md
├── episodes.json                      30 集索引（episode_index, sequence_id, object, task, frames, SOMA 反解误差）
├── frame_references/episode_0000NN.npz   打包提交时 --frame-references 用（见下）
└── reconstructed_data/<sequence_id>/  每集一个文件夹，可直接作为 soma_to_g1.py 的 data_folder
    ├── soma_params.npz                人体 SOMA-X 参数
    ├── poses.npy                      物体位姿 (T,4,4)
    ├── reconstructed_mesh/output_aligned.glb   物体网格（= 官方发布的网格，未改动）
    ├── ground_plane.json              官方发布的地面平面（原样复制）
    ├── object_pose_xyz_wxyz.npy       物体位姿 (T,7)，与数据集 observation.object.pose 同格式
    ├── camera.json                    我们估计的相机 K 与 T_world_cam
    └── episode.json                   本集元信息
```

`sequence_id` 与数据集 `meta/episodes_metadata.jsonl` 一致，`episode_index` 的对应关系见 `episodes.json`。

## 坐标系与格式约定

- **世界系**：官方发布的 OpenCV 世界系（x 右，y 下，z 前；重力沿 +y），与数据集 `observation.object.pose`、
  `ground_plane` 是同一个坐标系。单位米，30 fps，每集帧数与视频一致（每帧一行）。
- **`poses.npy`**：`(T,4,4)` float64，`world_T_object`，把 `reconstructed_mesh/output_aligned.glb` 的**网格文件坐标**
  变换到世界系。`soma_to_g1.py` 读入后会自己左乘 `diag(1,-1,-1,1)` 转到 SOMA 世界系，不需要预处理。
  注意：这里的 `output_aligned.glb` 就是官方网格本身（不是 CARI4D 的 oriented-bounds 对齐网格），两者配套使用。
- **`object_pose_xyz_wxyz.npy`**：`(T,7)`，`[x, y, z, qw, qx, qy, qz]`，与上面同一个位姿。
- **物体可见性**：所有帧都有位姿（遮挡帧来自跟踪），没有全零帧。
- **`soma_params.npz`**：官方 `save_soma_npz` 格式（`poses (T,77,3)` 局部旋转向量、`transl (T,3)` 米、
  `identity_coeffs (T,45)`、`scale_params (T,68)`、`bone_length_flexibles (T,6)`、`identity_model_type="mhr"`、
  `unit="meters"`、`keep_root=False` 等），坐标在 SOMA 原生轴（= OpenCV 世界系左乘 diag(1,-1,-1)），
  与数据集里人体参数的约定相同。加载时用 `SOMALayer(identity_model_type="mhr")`。
  MHR → SOMA 的反解误差每集记录在 `episode.json`（顶点误差均值约 0.4 cm）。
- **`camera.json`**：`K` 为针孔内参（1536×1152），`T_world_cam` 为相机到世界的 4×4。只供可视化/投影用。

## 怎么用

retarget（在 robotic_grounding 容器里）：

```bash
# 把一集放到 retarget 期望的位置（或直接把 data_folder 指向本目录）
cp -r reconstructed_data/<sequence_id> <repo>/robotic_grounding/reconstructed_data/
python scripts/retarget/soma_to_g1.py reconstructed_data/<sequence_id> --save
# 或全流程： scripts/retarget/process_soma_sequence.sh <sequence_id> --auto
```

`support_surface/<sequence_id>_support.usda` 本目录不重复提供，请从数据集 `track_2/tier_1_multiview_caption/support_surface/`
取，按数据集 README 放到 `<motion-root>/whole_body/reconstructed_stage/`。

打包 Tier 3 提交时，策略输入不是官方 Tier 3 镜像，所以要加：

```bash
python3 tools/pack_submission.py --track track_2_tier3 ... --frame-references /path/to/tier3_s1_20261020/frame_references
```

`frame_references/episode_0000NN.npz` 含 `episode_index`、`coordinate_frame="released_track2_world"`、
`frame_index [T]`、`pose_xyzw [T,1,7]`（位置米 + xyzw 四元数），即本目录的物体位姿。

## 已知问题（请在 retarget / 训练时注意）

- 质量参差很大，见下表。位置误差主要是**每集一个恒定的沿视线深度偏差**（±10 cm 量级），旋转误差主要来自近对称物体。
- **ep7（iron，熨斗）基本失败**：位置误差约 60 cm、旋转误差约 150°，建议不要用于训练，或单独处理。
- **ep12（black_platform）、ep15（wet_floor_sign）**：整集绕对称轴差 180°（画面上与真值几乎无法区分），旋转误差约 180°。
- ep2、3、4、9、13、17、19、27、28 等旋转误差大（几十度以上），多为对称物体的绕轴转角。
- 人体：第 0 帧与参考人体的质心差约 1–2 cm；整段人体是单目估计，可能有脚穿地/浮空几厘米、手与物体接触不精确。
  人体与物体用的是同一套相机和深度校正，相对关系是一致的，但物体还额外乘了 FoundationPose/CoCoNet 深度融合系数（每集一个常数，约 ±2%）。
- 规则说明：主办方允许用第 0 帧参考人体建立坐标系；我们还用它估计了相机主点，这一点属于规则灰色地带，尚未向主办方确认。

## 复现

代码在 `~/Projects/v2d/hoi_extraction/scripts/`（副本见本目录 `code/`），均在 `v2d_cari4d.sif` 容器内运行：

1. CARI4D 流程 + 后处理（P1/M1 → JC1 相机标定）
2. `postproc_joint_camera_K.py 0..29 --out-exp JC2s1_shared_f820_pp --sig-logf 1e-4 --sig-pp 0.01 --K-prior 820 758.1 560.5`
3. `postproc_depth_blend.py 0..29 --src JC2s1_shared_f820_pp --out-exp DB_JC2s1_shared_f820_pp`
4. `build_release.py <out> --variant DB_JC2s1_shared_f820_pp --camera JC2s1_shared_f820_pp`

评测：`eval_official.py --pipeline DB_JC2s1_shared_f820_pp --use-sim --align default=logs/JC2s1_shared_f820_pp_align.json`。

联系人：hanzhou

## 各集物体质量（官方 AUC，对照 Tier 2）

| ep | sequence_id | object | frames | AUC | 位置误差 cm | 旋转误差 ° | Tier 2 AUC | 超过 Tier 2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-06-04_14-16-23_tall_bar_stool_tip_up_04 | tall_bar_stool | 778 | 0.459 | 5.3 | 2.0 | 0.246 | 是 |
| 1 | 2026-05-18_17-06-21_vacuum_left_lean_02 | vacuum | 747 | 0.362 | 6.1 | 17.6 | 0.368 |  |
| 2 | 2026-06-02_10-04-42_blue_easter_basket_place_10 | blue_easter_basket | 980 | 0.027 | 19.1 | 103.6 | 0.799 |  |
| 3 | 2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05 | contigo_coffee_mug | 781 | 0.119 | 8.5 | 71.6 | 0.659 |  |
| 4 | 2026-04-24_11-40-52_baseball_bat_rotate_05 | baseball_bat | 803 | 0.108 | 9.4 | 80.6 | 0.526 |  |
| 5 | 2026-03-18_19-36-56_g1_box_rotate_01 | big_black_box | 1003 | 0.382 | 6.3 | 2.9 | 0.243 | 是 |
| 6 | 2026-06-01_11-26-44_monitor_rotate_05 | monitor | 546 | 0.145 | 9.1 | 10.3 | 0.414 |  |
| 7 | 2026-05-29_15-20-43_iron_round_table_02 | iron | 896 | 0.000 | 59.6 | 150.0 | 0.734 |  |
| 8 | 2026-06-05_15-08-08_skinny_wood_chair_push_10 | skinny_wood_chair | 708 | 0.730 | 3.0 | 2.1 | 0.206 | 是 |
| 9 | 2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03 | big_red_bowl | 870 | 0.016 | 9.5 | 82.3 | 0.609 |  |
| 10 | 2026-05-01_11-36-40_white_desk_lower_to_ground_10 | white_desk | 794 | 0.369 | 6.1 | 3.0 | 0.201 | 是 |
| 11 | 2026-04-24_13-56-58_light_blue_book_pick_place_09 | light_blue_book | 901 | 0.302 | 7.8 | 5.9 | 0.748 |  |
| 12 | 2026-04-25_17-20-04_black_platform_turn_05 | black_platform | 700 | 0.000 | 7.6 | 178.9 | 0.344 |  |
| 13 | 2026-06-09_10-19-09_snack_box_desk_through_06 | snack_box | 922 | 0.097 | 8.9 | 119.3 | 0.152 |  |
| 14 | 2026-05-28_14-44-09_round_table_flip_08 | round_table | 993 | 0.073 | 6.8 | 44.5 | 0.104 |  |
| 15 | 2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08 | wet_floor_sign | 715 | 0.000 | 4.1 | 177.5 | 0.433 |  |
| 16 | 2026-03-12_15-06-36_spray_bottle_left_desk_ground_02 | spray_bottle | 477 | 0.190 | 9.7 | 17.4 | 0.866 |  |
| 17 | 2026-05-01_17-04-55_traffic_cone_ground_table_02 | traffic_cone | 440 | 0.007 | 9.9 | 118.7 | 0.513 |  |
| 18 | 2026-05-06_10-00-19_blue_trash_can_drag_sideways_08 | blue_trash_can | 858 | 0.010 | 14.3 | 4.8 | 0.069 |  |
| 19 | 2026-05-19_10-19-48_cane_swing_05 | cane | 886 | 0.002 | 15.8 | 91.4 | 0.354 |  |
| 20 | 2026-06-08_16-57-34_rice_cooker_clockwise_04 | rice_cooker | 553 | 0.015 | 12.9 | 2.0 | 0.634 |  |
| 21 | 2026-04-29_11-53-57_tall_bar_stool_pull_08 | tall_bar_stool | 678 | 0.527 | 4.7 | 2.3 | 0.138 | 是 |
| 22 | 2026-05-22_16-52-36_step_ladder_drag_03 | step_ladder | 719 | 0.239 | 7.8 | 1.7 | 0.288 |  |
| 23 | 2026-06-03_15-08-03_toy_stroller_back_forth_ground_04 | toy_stroller | 946 | 0.358 | 6.4 | 2.9 | 0.574 |  |
| 24 | 2026-05-29_15-43-20_watering_can_pour_right_02 | watering_can | 748 | 0.450 | 5.9 | 3.5 | 0.730 |  |
| 25 | 2026-03-20_11-59-57_cyan_water_bottle_rotate_01 | cyan_water_bottle | 859 | 0.058 | 10.5 | 43.2 | 0.796 |  |
| 26 | 2026-06-05_16-05-08_skateboard_rotate_ground_10 | skateboard | 669 | 0.533 | 4.6 | 3.4 | 0.247 | 是 |
| 27 | 2026-05-01_11-15-05_woven_basket_ground_desk_10 | woven_basket | 565 | 0.063 | 8.4 | 74.7 | 0.714 |  |
| 28 | 2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_ | squeegee | 988 | 0.020 | 19.2 | 77.8 | 0.135 |  |
| 29 | 2026-06-05_18-27-40_rolling_luggage_left_foot_push_06 | rolling_luggage | 467 | 0.651 | 3.5 | 3.7 | 0.135 | 是 |
