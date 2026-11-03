# choices: per-episode choice (own-input numbers only)

Own input = each packed rollout against its own Tier-3 training input (the official evaluator's numbers in the packed results JSON). No clean number appears here.

| ep | object | item | run | own AUC | SP | MP | MPPE cm | arm mean AUC | round-1 own AUC | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 00 | tall_bar_stool | f0 | 4 | 0.9657 | 1 | 1 | 0.53 | 0.9657 | 0.9657 | not a retry candidate (own-input SP = MP = 1) |
| 01 | vacuum | f0 | 3 | 0.1990 | 0 | 0 | 32.82 | 0.1989 | 0.1990 | KEEP_F0: retry both 0/5 AUC 0.1864 SP 0 MP 0 vs f0 final both 0/5 AUC  |
| 02 | blue_easter_basket | retry | 5 | 0.3239 | 0 | 0 | 13.96 | 0.3254 | 0.1790 | REPLACE (ii): AUC 0.3254 >= 0.1803 + 0.01, SP 1>=0, MP 0>=0 |
| 03 | contigo_coffee_mug | f0 | 4 | 0.9864 | 1 | 1 | 0.62 | 0.9865 | 0.9864 | not a retry candidate (own-input SP = MP = 1) |
| 04 | baseball_bat | retry | 2 | 0.1842 | 0 | 0 | 126.01 | 0.1840 | 0.1708 | REPLACE (ii): AUC 0.1840 >= 0.1707 + 0.01, SP 0>=0, MP 0>=0 |
| 05 | big_black_box | retry | 5 | 0.5548 | 1 | 1 | 2.23 | 0.5552 | 0.2841 | REPLACE (i): retry SP=MP=1 in 5/5 vs f0 final 0/5 |
| 06 | monitor | f0 | 3 | 0.8498 | 1 | 1 | 1.85 | 0.8486 | 0.8498 | not a retry candidate (own-input SP = MP = 1) |
| 07 | iron | f0 | 1 | 0.9867 | 1 | 1 | 0.69 | 0.9867 | 0.9867 | not a retry candidate (own-input SP = MP = 1) |
| 08 | skinny_wood_chair | f0 | 1 | 0.9628 | 1 | 1 | 0.96 | 0.9629 | 0.9628 | not a retry candidate (own-input SP = MP = 1) |
| 09 | big_red_bowl | f0 | 5 | 0.9227 | 1 | 1 | 1.32 | 0.9231 | 0.9227 | not a retry candidate (own-input SP = MP = 1) |
| 10 | white_desk | f0 | 4 | 0.5521 | 1 | 1 | 1.74 | 0.5492 | 0.5521 | not a retry candidate (own-input SP = MP = 1) |
| 11 | light_blue_book | f0 | 3 | 0.1004 | 0 | 0 | 73.29 | 0.1006 | 0.1004 | KEEP_F0: retry both 0/5 AUC 0.1032 SP 0 MP 0 vs f0 final both 0/5 AUC  |
| 12 | black_platform | f0 | 1 | 0.7042 | 1 | 1 | 2.24 | 0.7041 | 0.7042 | not a retry candidate (own-input SP = MP = 1) |
| 13 | snack_box | retry | 2 | 0.2099 | 0 | 0 | 61.36 | 0.2099 | 0.1953 | REPLACE (ii): AUC 0.2099 >= 0.1951 + 0.01, SP 0>=0, MP 0>=0 |
| 14 | round_table | retry | 2 | 0.5889 | 1 | 0 | 9.51 | 0.5883 | 0.5442 | REPLACE (ii): AUC 0.5883 >= 0.5455 + 0.01, SP 5>=0, MP 0>=0 |
| 15 | wet_floor_sign | f0 | 2 | 0.8897 | 1 | 1 | 0.90 | 0.8898 | 0.8897 | not a retry candidate (own-input SP = MP = 1) |
| 16 | spray_bottle | retry | 5 | 0.5929 | 0 | 0 | 9.23 | 0.5964 | 0.3405 | REPLACE (ii): AUC 0.5964 >= 0.3439 + 0.01, SP 0>=0, MP 0>=0 |
| 17 | traffic_cone | retry | 1 | 0.9432 | 1 | 1 | 0.97 | 0.9434 | 0.2761 | REPLACE (i): retry SP=MP=1 in 5/5 vs f0 final 0/5 |
| 18 | blue_trash_can | f0 | 2 | 0.9718 | 1 | 1 | 0.61 | 0.9720 | 0.9718 | not a retry candidate (own-input SP = MP = 1) |
| 19 | cane | retry | 1 | 0.1568 | 0 | 0 | 80.95 | 0.1568 | 0.1304 | REPLACE (ii): AUC 0.1568 >= 0.1304 + 0.01, SP 0>=0, MP 0>=0 |
| 20 | rice_cooker | f0 | 1 | 0.9174 | 1 | 1 | 0.89 | 0.9182 | 0.9174 | not a retry candidate (own-input SP = MP = 1) |
| 21 | tall_bar_stool | retry | 5 | 0.5060 | 1 | 1 | 2.80 | 0.5047 | 0.3779 | REPLACE (i): retry SP=MP=1 in 5/5 vs f0 final 0/5 |
| 22 | step_ladder | f0 | 1 | 0.9589 | 1 | 1 | 0.65 | 0.9589 | 0.9589 | not a retry candidate (own-input SP = MP = 1) |
| 23 | toy_stroller | retry | 5 | 0.9782 | 1 | 1 | 0.62 | 0.9780 | 0.5009 | REPLACE (i): retry SP=MP=1 in 5/5 vs f0 final 0/5 |
| 24 | watering_can | f0 | 2 | 0.9285 | 1 | 1 | 1.01 | 0.9294 | 0.9285 | not a retry candidate (own-input SP = MP = 1) |
| 25 | cyan_water_bottle | retry | 5 | 0.5835 | 1 | 1 | 3.47 | 0.5849 | 0.1484 | REPLACE (i): retry SP=MP=1 in 5/5 vs f0 final 0/5 |
| 26 | skateboard | f0 | 1 | 0.9670 | 1 | 1 | 0.45 | 0.9671 | 0.9670 | not a retry candidate (own-input SP = MP = 1) |
| 27 | woven_basket | f0 | 1 | 0.8723 | 1 | 1 | 1.12 | 0.8723 | 0.8723 | not a retry candidate (own-input SP = MP = 1) |
| 28 | squeegee | f0 | 3 | 0.1403 | 0 | 0 | 56.67 | 0.1392 | 0.1403 | KEEP_F0: retry both 0/5 AUC 0.1394 SP 0 MP 0 vs f0 final both 0/5 AUC  |
| 29 | rolling_luggage | f0 | 5 | 0.7810 | 1 | 1 | 2.34 | 0.7815 | 0.7810 | not a retry candidate (own-input SP = MP = 1) |

| totals (30 episodes, own input) | mean AUC | SP-SR = 1 | MP-SR = 1 | mean MPPE cm |
|---|---|---|---|---|
| choices | 0.6760 | 22 | 21 | 16.39 |
| round-1 FINAL_f0 (all 30 episodes on their t3_f0_final items) | 0.5935 | 17 | 16 | 24.57 |
