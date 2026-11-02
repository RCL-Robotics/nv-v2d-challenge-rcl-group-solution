"""Support-surface reconstruction: shared settings. The support class of each episode comes from its task text
(inputs/task.txt); its scanned mesh is the dataset mesh of the same furniture (used as an object in other episodes)."""
import os
V = os.environ["V2D_ROOT"]; OUT = os.environ.get("V2D_SUPPORT_OUT", f"{V}/support")
MESH = f"{V}/dataset/track_2/tier_1_multiview_caption/mesh"
SUPPORT = {1: "tall_bar_stool", 2: "round_table", 3: "round_table", 6: "round_table", 7: "round_table", 9: "skinny_wood_chair",
           11: "white_desk", 13: "white_desk", 16: "white_desk", 17: "round_table", 20: "round_table", 25: "round_table",
           27: "white_desk", 28: "tall_bar_stool"}
PROMPT = {"round_table": "round table.", "tall_bar_stool": "bar stool.", "skinny_wood_chair": "chair.", "white_desk": "white desk."}
def inp(i): return f"{V}/agent_runs/T1_ds_ep{i:03d}/inputs"
