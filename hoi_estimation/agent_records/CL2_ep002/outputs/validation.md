| phase | frames | vis | IoU base | IoU ours | centroid err px base/ours | texture z base/ours | lug cos^2 base/ours | median dR deg/frame base/ours | median acc cm/f^2 base/ours | floor height cm base/ours | tilt vs floor deg base/ours | hand dist cm base/ours | dz mean/median cm (ours-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_table_static | 0-124 | 125 | 0.85 | 0.92 | 3.27/0.39 | 0.23/2.08 | 0.51/0.00 | 0.13/0.00 | 0.04/0.00 | 81.82/81.91 | 0.84/0.00 | 78.97/78.10 | -0.41/0.00 |
| C1_carry | 125-213 | 86 | 0.75 | 0.85 | 2.64/0.77 | -0.06/0.39 | -/- | 6.67/1.14 | 2.64/0.19 | - | 8.14/14.76 | 15.88/14.63 | -0.92/-0.55 |
| S2_floor_static | 214-436 | 223 | 0.80 | 0.85 | 3.75/0.59 | 0.13/-0.29 | 0.77/0.23 | 0.00/0.00 | 0.10/0.00 | 1.08/0.08 | 2.59/0.00 | 61.44/61.95 | 0.24/0.00 |
| C2_carry_handover | 437-846 | 304 | 0.64 | 0.89 | 6.61/0.60 | 0.02/0.52 | 0.34/0.02 | 3.98/0.40 | 2.47/0.12 | - | 6.35/4.34 | 12.21/14.74 | -19.91/0.75 |
| S3_floor_static_occluded | 847-979 | 20 | 0.45 | 0.89 | 5.97/0.52 | -/- | -/- | 5.62/0.00 | 5.17/0.00 | 5.27/-7.67 | 12.05/0.00 | 73.04/83.85 | 0.33/0.00 |
| ALL | 0-979 | 758 | 0.73 | 0.88 | 3.64/0.54 | 0.08/0.53 | 0.55/0.10 | 1.48/0.05 | 1.82/0.03 | - | 3.95/0.44 | 19.98/19.74 | -8.37/0.16 |

S1_table_static: yaw base circmean 65 (circ std 10), ours 294 (circ std 0); max dR/frame base 136.1 ours 0.0; ours-base rot diff median 131 deg, pos diff median 1.0 cm
C1_carry: yaw base circmean 273 (circ std 35), ours 271 (circ std 9); max dR/frame base 35.3 ours 5.8; ours-base rot diff median 28 deg, pos diff median 2.2 cm
  f"(circ std {np.degrees(np.sqrt(-2*np.log(abs(np.exp(1j*np.radians(mr['yaw'][sl])).mean())))):.0f}); "
S2_floor_static: yaw base circmean 219 (circ std 19), ours 267 (circ std nan); max dR/frame base 58.2 ours 5.3; ours-base rot diff median 40 deg, pos diff median 2.6 cm
C2_carry_handover: yaw base circmean 293 (circ std 59), ours 289 (circ std 9); max dR/frame base 159.0 ours 10.4; ours-base rot diff median 43 deg, pos diff median 5.2 cm
S3_floor_static_occluded: yaw base circmean 93 (circ std 63), ours 291 (circ std 0); max dR/frame base 78.8 ours 0.1; ours-base rot diff median 168 deg, pos diff median 15.4 cm

S1_table_static: lowest point 81.9 cm above the released floor plane -> floor-contact cue: move +539.8 cm along the viewing ray (ray . up = -0.152); (floor plane tolerance from foot fit: median 4.0 cm)
S2_floor_static: lowest point 0.0 cm above the released floor plane -> floor-contact cue: move +0.1 cm along the viewing ray (ray . up = -0.329); (floor plane tolerance from foot fit: median 4.0 cm)
S3_floor_static_occluded: lowest point -7.7 cm above the released floor plane -> floor-contact cue: move -18.7 cm along the viewing ray (ray . up = -0.413); (floor plane tolerance from foot fit: median 4.0 cm)
