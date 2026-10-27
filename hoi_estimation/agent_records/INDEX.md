# Agent 运行记录索引

每个目录：`AGENT_TASK.md`（任务说明）、`transcript.md`（可读对话记录）、`events.jsonl.gz`（原始事件流，去掉了图片和签名）、`run_manifest.txt`（环境、输入哈希、耗时）、`outputs/`（报告、求解脚本、检查图，不含大数组）。

| run | 系列 | 模型 | episode | 日期 | 耗时(s) | 花费(USD) | 有报告 | 报告标题 |
|---|---|---|---|---|---|---|---|---|
| A1_codex_ep003 | Codex GPT 修正轨迹，第一版任务 | gpt-6-astra | 003 | 2026-10-18 | 3102 |  | 是 | Object trajectory refinement |
| A2_codex_ep013 | Codex GPT 修正轨迹 | gpt-6-astra | 013 | 2026-10-18 | 2113 |  | 是 | Object trajectory result |
| A3_codex_ep004 | Codex GPT 修正轨迹，开放网络 | gpt-6-astra | 004 | 2026-10-18 | 2423 |  | 是 | The result is `object_poses_world.npz`, containing `T_world_ …[+254 chars] |
| A3_codex_ep009 | Codex GPT 修正轨迹，开放网络 | gpt-6-astra | 009 | 2026-10-18 | 1419 |  | 是 | The output is `object_poses_world.npz`, containing `T_world_ …[+150 chars] |
| A3_codex_ep012 | Codex GPT 修正轨迹，开放网络 | gpt-6-astra | 012 | 2026-10-18 | 939 |  | 是 | Platform trajectory refinement |
| A3_codex_ep015 | Codex GPT 修正轨迹，开放网络 | gpt-6-astra | 015 | 2026-10-18 | 767 |  | 是 | Object trajectory refinement |
| A3_codex_ep027 | Codex GPT 修正轨迹，开放网络 | gpt-6-astra | 027 | 2026-10-18 | 550 |  | 是 | Basket trajectory refinement |
| A4_codex_ep002 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 002 | 2026-10-19 | 727 |  | 是 | Object trajectory refinement |
| A4_codex_ep003 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 003 | 2026-10-19 | 1948 |  | 是 | Monocular mug trajectory repair |
| A4_codex_ep014 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 014 | 2026-10-19 | 2929 |  | 是 | Round-table trajectory refinement |
| A4_codex_ep017 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 017 | 2026-10-19 | 1699 |  |  |  |
| A4_codex_ep019 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 019 | 2026-10-19 | 3696 |  | 是 | Cane trajectory result |
| A4_codex_ep025 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 025 | 2026-10-19 | 2887 |  | 是 | Object trajectory refinement report |
| A4_codex_ep028 | Codex GPT 时序修正（temporal 任务） | gpt-6-astra | 028 | 2026-10-19 | 3687 |  | 是 | Squeegee trajectory correction |
| C1_ds_ep000 | Codex DeepSeek 估计相机 | deepseek-flash | 000 | 2026-10-20 | 1571 |  | 是 | Camera and world-trajectory estimate |
| C1_ds_ep018 | Codex DeepSeek 估计相机 | deepseek-flash | 018 | 2026-10-20 | 2357 |  | 是 | Camera and world trajectory estimate |
| CL1_ep002 | Claude Opus 完整轨迹，第一批 | claude-opus-5-5 | 002 | 2026-10-20 | 4839 | 12.86 | 是 | Episode 2 (blue easter basket) – trajectory-level object pos …[+10 chars] |
| CL1_ep003 | Claude Opus 完整轨迹，第一批 | claude-opus-5-5 | 003 | 2026-10-20 | 3549 | 7.11 | 是 | Mug trajectory, episode 3 (`contigo_coffee_mug`, 781 frames) …[+8 chars] |
| CL1_ep010 | Claude Opus 完整轨迹，第一批 | claude-opus-5-5 | 010 | 2026-10-20 | 3570 | 4.78 | 是 | White desk (episode 10): object trajectory recovery |
| CL1_ep011 | Claude Opus 完整轨迹，第一批 | claude-opus-5-5 | 011 | 2026-10-20 | 2967 | 6.81 | 是 | Episode 11 — light_blue_book: trajectory-level object pose r …[+7 chars] |
| CL1_ep024 | Claude Opus 完整轨迹，第一批 | claude-opus-5-5 | 024 | 2026-10-20 | 2380 | 4.24 | 是 | Episode 24 – watering can: trajectory-level object pose reco …[+4 chars] |
| CL2_ep000 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 000 | 2026-10-21 | 3578 | 5.44 | 是 | Episode 0 – tall bar stool: object trajectory re-estimation |
| CL2_ep001 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 001 | 2026-10-21 | 3201 | 5.06 | 是 | Vacuum episode: object trajectory report |
| CL2_ep002 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 002 | 2026-10-21 | 3767 | 8.70 | 是 | Episode 2 – blue easter basket: trajectory-level object pose …[+14 chars] |
| CL2_ep003 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 003 | 2026-10-21 | 3676 | 8.22 | 是 | Episode 3: contigo coffee mug, 781 frames. Trajectory re-est …[+14 chars] |
| CL2_ep004 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 004 | 2026-10-21 | 5262 | 9.54 | 是 | Bat trajectory, episode 4: report |
| CL2_ep005 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 005 | 2026-10-21 | 2300 | 4.80 | 是 | Episode 5 (big_black_box): object trajectory recovery |
| CL2_ep006 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 006 | 2026-10-21 | 3721 | 7.48 | 是 | Episode 6: monitor on a round table, 6-DoF object trajectory |
| CL2_ep007 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 007 | 2026-10-21 | 3622 | 8.53 | 是 | Iron trajectory, episode 7: report |
| CL2_ep008 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 008 | 2026-10-21 | 1933 | 3.46 | 是 | Episode 8: skinny_wood_chair ("walk to the chair and push it …[+23 chars] |
| CL2_ep009 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 009 | 2026-10-21 | 4043 | 8.92 | 是 | Episode 9 – big red bowl: trajectory-level 6-DoF recovery |
| CL2_ep010 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 010 | 2026-10-21 | 3860 | 5.21 | 是 | White desk (episode 10): trajectory-level object pose recove …[+2 chars] |
| CL2_ep011 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 011 | 2026-10-21 | 4881 | 8.13 |  |  |
| CL2_ep012 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 012 | 2026-10-21 | 1758 | 3.03 |  |  |
| CL2_ep013 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 013 | 2026-10-21 | 1252 | 2.42 |  |  |
| CL2_ep014 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 014 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep015 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 015 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep016 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 016 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep017 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 017 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep018 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 018 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep019 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 019 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep020 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 020 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep021 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 021 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep022 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 022 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep023 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 023 | 2026-10-21 | 2 | 0 |  |  |
| CL2_ep024 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 024 | 2026-10-21 | 2 | 0 |  |  |
| CL2_ep025 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 025 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep026 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 026 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep027 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 027 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep028 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 028 | 2026-10-21 | 3 | 0 |  |  |
| CL2_ep029 | Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10） | claude-opus-5-5 | 029 | 2026-10-21 | 3 | 0 |  |  |
| CL2b_ep011 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 011 | 2026-10-21 | 4715 | 7.65 | 是 | Episode 11 (light_blue_book): trajectory-level object pose r …[+7 chars] |
| CL2b_ep012 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 012 | 2026-10-21 | 3041 | 4.42 | 是 | Episode 12 – black_platform (a plastic pallet): object traje …[+5 chars] |
| CL2b_ep013 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 013 | 2026-10-21 | 3627 | 7.10 | 是 | Episode 13 (snack_box): trajectory-level object pose re-esti …[+6 chars] |
| CL2b_ep014 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 014 | 2026-10-21 | 5387 | 12.99 | 是 | Round table, episode 14: trajectory re-estimation report |
| CL2b_ep015 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 015 | 2026-10-21 | 3548 | 6.50 | 是 | Episode 15 — wet floor sign: trajectory-level object pose re …[+6 chars] |
| CL2b_ep016 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 016 | 2026-10-21 | 3237 | 6.34 | 是 | Episode 16 – spray bottle: trajectory-level re-estimation of …[+16 chars] |
| CL2b_ep017 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 017 | 2026-10-21 | 2849 | 6.16 | 是 | Traffic-cone trajectory (episode 17): report |
| CL2b_ep018 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 018 | 2026-10-21 | 2547 | 4.48 | 是 | Episode 18: blue trash can, trajectory-level pose re-estimat …[+3 chars] |
| CL2b_ep019 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 019 | 2026-10-21 | 3876 | 4.95 | 是 | Episode 19: cane (quad cane): 6-DoF trajectory recovery |
| CL2b_ep020 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 020 | 2026-10-21 | 2574 | 5.58 | 是 | Rice cooker, episode 20: object trajectory report |
| CL2b_ep021 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 021 | 2026-10-21 | 2607 | 4.11 | 是 | Episode 21 – tall bar stool: trajectory refinement |
| CL2b_ep022 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 022 | 2026-10-21 | 1053 | 2.00 |  |  |
| CL2b_ep023 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 023 | 2026-10-21 | 737 | 1.63 |  |  |
| CL2b_ep024 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 024 | 2026-10-21 | 177 | 0.54 |  |  |
| CL2b_ep025 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 025 | 2026-10-21 | 3 | 0 |  |  |
| CL2b_ep026 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 026 | 2026-10-21 | 3 | 0 |  |  |
| CL2b_ep027 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 027 | 2026-10-21 | 3 | 0 |  |  |
| CL2b_ep028 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 028 | 2026-10-21 | 3 | 0 |  |  |
| CL2b_ep029 | Claude Opus v2（ep11-21） | claude-opus-5-5 | 029 | 2026-10-21 | 3 | 0 |  |  |
| CL2c_ep022 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 022 | 2026-10-21 | 4033 | 4.89 | 是 | Episode 22 — step ladder, dragged on the floor: trajectory-l …[+16 chars] |
| CL2c_ep023 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 023 | 2026-10-21 | 2721 | 5.28 | 是 | Episode 23 – toy stroller: trajectory recovery report |
| CL2c_ep024 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 024 | 2026-10-21 | 3117 | 5.56 | 是 | Episode 24 – watering can: object trajectory report |
| CL2c_ep025 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 025 | 2026-10-21 | 3018 | 6.10 | 是 | Episode 25 – cyan water bottle: trajectory-level 6-DoF recov …[+3 chars] |
| CL2c_ep026 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 026 | 2026-10-21 | 3567 | 5.13 | 是 | Episode 26: skateboard rotated on the ground. Trajectory-lev …[+21 chars] |
| CL2c_ep027 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 027 | 2026-10-21 | 4242 | 9.17 | 是 | Episode 27 (woven_basket): trajectory-level object pose |
| CL2c_ep028 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 028 | 2026-10-21 | 5156 | 8.47 | 是 | Squeegee episode 28: trajectory-level object pose recovery |
| CL2c_ep029 | Claude Opus v2（ep22-29） | claude-opus-5-5 | 029 | 2026-10-21 | 2450 | 4.26 | 是 | Episode 29: rolling luggage, pushed with the left foot. Traj …[+20 chars] |
| D1_codex_ep020 | Codex GPT 修正深度/位置 | gpt-6-astra | 020 | 2026-10-19 | 803 |  | 是 | Fixed-rotation translation refinement |
| T1_ds_ep000 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 000 | 2026-10-20 | 2406 |  | 是 | Tall bar stool trajectory recovery |
| T1_ds_ep001 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 001 | 2026-10-20 | 3459 |  | 是 | Vacuum 6-DoF Trajectory Recovery |
| T1_ds_ep002 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 002 | 2026-10-20 | 3590 |  | 是 | Monocular 6-DoF Basket Trajectory Recovery |
| T1_ds_ep003 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 003 | 2026-10-20 | 4262 |  | 是 | Monocular rigid-object trajectory recovery |
| T1_ds_ep004 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 004 | 2026-10-20 | 5306 |  | 是 | Baseball-bat 6-DoF trajectory recovery |
| T1_ds_ep005 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 005 | 2026-10-20 | 1786 |  | 是 | Big Black Box: World-Frame Trajectory Recovery |
| T1_ds_ep006 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 006 | 2026-10-20 | 4589 |  | 是 | Monitor 6-DoF trajectory recovery |
| T1_ds_ep007 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 007 | 2026-10-20 | 4108 |  | 是 | Monocular iron trajectory recovery |
| T1_ds_ep008 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 008 | 2026-10-20 | 2701 |  | 是 | 6-DoF Skinny Wood Chair Trajectory |
| T1_ds_ep009 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 009 | 2026-10-20 | 1231 |  | 是 | Bowl Trajectory Recovery Report |
| T1_ds_ep010 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 010 | 2026-10-20 | 1207 |  | 是 | White-desk trajectory recovery |
| T1_ds_ep011 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 011 | 2026-10-20 | 5401 |  |  |  |
| T1_ds_ep012 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 012 | 2026-10-20 | 4344 |  | 是 | Rigid object trajectory reconstruction |
| T1_ds_ep013 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 013 | 2026-10-20 | 5401 |  |  |  |
| T1_ds_ep014 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 014 | 2026-10-20 | 3143 |  | 是 | Round-table 6-DoF trajectory recovery |
| T1_ds_ep015 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 015 | 2026-10-20 | 1545 |  | 是 | Object trajectory recovery report |
| T1_ds_ep016 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 016 | 2026-10-20 | 1821 |  | 是 | Spray-bottle 6-DoF trajectory recovery |
| T1_ds_ep017 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 017 | 2026-10-20 | 1978 |  | 是 | Traffic-cone 6-DoF trajectory recovery |
| T1_ds_ep018 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 018 | 2026-10-20 | 3987 |  | 是 | Blue Trash-Can Trajectory Recovery |
| T1_ds_ep019 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 019 | 2026-10-20 | 5400 |  |  |  |
| T1_ds_ep020 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 020 | 2026-10-20 | 1330 |  | 是 | Rice-cooker trajectory recovery |
| T1_ds_ep021 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 021 | 2026-10-20 | 1628 |  | 是 | Tall bar stool trajectory recovery |
| T1_ds_ep022 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 022 | 2026-10-20 | 2196 |  | 是 | Step-ladder 6-DoF trajectory recovery |
| T1_ds_ep023 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 023 | 2026-10-20 | 3748 |  | 是 | Toy-stroller 6-DoF trajectory recovery |
| T1_ds_ep024 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 024 | 2026-10-20 | 3078 |  | 是 | Watering-can 6-DoF trajectory — method and validation |
| T1_ds_ep025 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 025 | 2026-10-20 | 5121 |  | 是 | Monocular 6-DoF Bottle Trajectory |
| T1_ds_ep026 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 026 | 2026-10-20 | 2605 |  | 是 | Skateboard 6-DoF trajectory recovery |
| T1_ds_ep027 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 027 | 2026-10-20 | 2070 |  | 是 | Woven-basket 6-DoF trajectory recovery |
| T1_ds_ep028 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 028 | 2026-10-20 | 4831 |  | 是 | Squeegee 6-DoF Trajectory Report |
| T1_ds_ep029 | Codex DeepSeek 完整轨迹（Tier 1 输入） | deepseek-flash | 029 | 2026-10-20 | 2091 |  | 是 | Rolling-luggage trajectory recovery |
| smoke | 冒烟测试 | gpt-6-astra | - | 2026-10-18 | 43 |  | 是 | Status: blocked by missing input data |
| smoke_claude | 冒烟测试 | claude-opus-5-5 | - | 2026-10-20 | 16 | 0.14 |  |  |
| smoke_ds | 冒烟测试 | deepseek-flash | - | 2026-10-20 | 8 |  |  |  |
