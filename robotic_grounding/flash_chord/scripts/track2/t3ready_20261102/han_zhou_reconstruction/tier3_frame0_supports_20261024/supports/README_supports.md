# Tier 3 支撑面重建（14 个 episode）

物体会被放在桌子、椅子或台子上的 14 个 episode：ep01、02、03、06、07、09、11、13、16、17、20、25、27、28。这里提供这些支撑物在世界坐标系中的重建结果。世界坐标系与 Tier 3 发布包相同（OpenCV world，y 轴朝下）；地面平面见各 episode 输入中的 `ground_plane.json`。

## 支撑物与对应 episode

| 支撑物 | episode | 承载面高度 |
|---|---|---|
| round_table | 02 03 06 07 17 20 25 | 0.786 m |
| white_desk | 11 13 16 27 | 0.914 m |
| tall_bar_stool | 01 28（物体斜靠在吧凳上） | 座面 0.634 m |
| skinny_wood_chair | 09 | 座面 0.514 m |

## 每个 episode 目录（episode_XXXXXX/）中的文件

- `support_mesh.glb`：家具的扫描网格。它就是数据集里同一件家具的网格（这些家具在其他 episode 里作为操作物体出现过）。
- `support.json`，主要字段：
  - `T_world_mesh`：4×4 变换，把 `support_mesh.glb` 放到世界坐标系中。这是视频拟合得到的位姿。
  - `T_world_mesh_contact`：在 `T_world_mesh` 基础上，沿竖直方向平移 `contact_shift_m`（−5 到 +2 cm），使我们的物体轨迹静止时正好落在支撑面上。**仿真推荐用这个。** 平移后家具腿会略微插入地面或离开地面，作为固定物体没有影响。
  - `top_box`：承载面（桌面或座面）的有向盒子近似，3 cm 厚。给出中心、三个轴（按列）和半长，可以直接作为碰撞体。
  - `support_surface_height_m`：承载面高度。
  - `iou`：拟合后剪影与分割掩码的 IoU。
  - `up_world`：世界坐标系中的竖直向上方向。
- `support_world.obj`：简化后的网格，已经变换到世界坐标系，用于预览。简化后有漏洞，做碰撞时建议用 glb 加位姿。
- `object_poses_world.npz`：本 episode 的物体轨迹，也就是目前最好的 Tier 3 结果（Opus v2 加第一帧修正），未作改动。
- `fit_overlay.jpg`：拟合检查图。红色为 SAM2 分割，绿色轮廓为投影后的网格，黄框为 GroundingDINO 检测框。

## 方法

1. 根据任务文本确定支撑物是哪件家具（任务文本属于输入）。
2. 对视频取时序中值，得到去掉人的背景图。在背景图上用 GroundingDINO 检测家具，取离物体静止位置最近的检测框，再用 SAM2 分割。
3. 约束家具立在已知地面上（最低点贴地），在 6 个可能的朝上方向、偏航角和地面平面内的位置上做网格搜索。然后用 nvdiffrast 渲染剪影，与分割掩码做 IoU，用梯度精修。过程中不使用任何 GT。

## 质量

- **拟合精度**：剪影 IoU 在 0.61 到 0.82 之间。
- **ep01 精度较低**：IoU 0.61，吧凳的朝向和位置有误差。好在这一集吸尘器只是斜靠在吧凳上，影响有限。
- **GT 诊断（只用于评估，不参与生成）**：把 GT 物体位姿放在我们重建的支撑面上，静止时物体底面和承载面的偏差在 −2.9 到 +1.9 cm 之间，说明承载面高度的误差大约在 ±2–3 cm。
- **ep17 偏差较大**：用我们自己的轨迹算是 −4.9 cm（物体插入桌面），而 GT 只有 −0.6 cm，差别来自锥桶的旋转误差。
- **斜靠的两集（GT 诊断）**：ep01 吸尘器开头离吧凳 3.4 cm，和 GT 一致。ep28 结束时我们的轨迹离吧凳 10 cm，而 GT 是 0 cm，即贴着吧凳。

代码：`~/Projects/v2d/support/`。依次运行 background、anchor、detect、segment、fit、contact、package。
