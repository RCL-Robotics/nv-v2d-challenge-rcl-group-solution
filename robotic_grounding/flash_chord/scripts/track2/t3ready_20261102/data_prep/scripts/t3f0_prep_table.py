#!/usr/bin/env python
"""t3f0_prep_table.py -- one row per episode of the Tier-3 f0 prep (build report, retarget checks, supports, physical
checks) -> tmp_analysis/tier3/f0/build/prep_table.{json,tsv}. Read-only on everything else. No clean data."""
import json
from pathlib import Path

ME = Path("/root/autodl-tmp/haoran.zhang")
A = ME / "tmp_analysis/tier3/f0"
HAN_FLAGS = {1: "Han: stool fit IoU 0.61", 17: "Han: cone track ~5 cm inside the table (table lowered)",
             28: "Han: squeegee ends ~10 cm from the stool"}


def main():
    br = json.loads((A / "build/build_report.json").read_text())
    phys = json.loads((A / "physcheck/physcheck_summary.json").read_text())["episodes"]
    rows = []
    for k, e in br["episodes"].items():
        i = int(k)
        rc = json.loads((A / "retarget" / f"ep{k}_rcheck.json").read_text())
        vo = rc.get("vs_opus_parquet", {})
        lift = rc.get("retarget_object_ground_correction") or ""
        row = dict(ep=i, object=e["object"], frames=e["frames"],
                   track_change_vs_opus=f"{e['vs_opus_v2']['pos_change_cm_median_max'][0]} cm / {e['vs_opus_v2']['rot_change_deg_median_max'][0]} deg",
                   retarget="rerun (object enters object_*, hand_contact_active, frame-0 object ground lift)",
                   rcheck_all_ok=rc["all_ok"], rcheck_failed=[g for g, v in rc["gates"].items() if not v["ok"]],
                   robot_max_diff_vs_opus=max(vo.get("robot_max_abs_diff", {"-": float("nan")}).values()),
                   ik_iter_diff=vo.get("ik_num_iterations_max_diff_frames"),
                   hand_contact_frames_LR_f0=vo.get("hand_contact_frames_f0_left_right"),
                   hand_contact_frames_LR_opus=vo.get("hand_contact_frames_opus_left_right"),
                   object_ground_lift=lift.split("applied=")[1].split(" ")[0].rstrip(",") if "applied=" in lift else lift,
                   parquet_sha256=rc.get("parquet_sha256"), flags=[])
        if i in HAN_FLAGS:
            row["flags"].append(HAN_FLAGS[i])
        sp = A / "supports" / f"ep{k}.json"
        if sp.is_file():
            s = json.loads(sp.read_text())
            p = phys[k]
            row["support"] = s["support"]
            if s["kind"] == "top":
                sh = s["shape"]
                geo = (f"disk r {sh['radius_m']:.3f} m" if sh["kind"] == "disk" else
                       f"rect {2 * sh['half_extents_m'][0]:.2f} x {2 * sh['half_extents_m'][1]:.2f} m yaw {sh['yaw_deg']:.0f}")
                row["support_geometry"] = (f"{geo}, top z {s['z_top_m']:.3f} m, 3 cm slab, {s['n_boxes']} strips "
                                           f"(area {100 * s['strip_area_fraction']:.0f}%)")
                row["support_pose"] = (f"Han contact shift {s['han_contact_shift_cm']:+.2f} cm, refit {s['refit_residual_cm']:+.2f} cm "
                                       f"(anchor run {s['anchor_run']}), net vs video fit {s['total_shift_vs_videofit_cm']:+.2f} cm"
                                       + (f", footprint shift {s['anchor_footprint']['shift_xy_cm']} cm" if s["anchor_footprint"]["applied"] else ""))
            else:
                row["support_geometry"] = f"stool contact patches: {s['n_boxes']} boxes ({100 * s['voxel_m']:.0f} cm voxels, carved {s['n_voxels_carved']})"
                row["support_pose"] = (f"T_world_mesh_contact (shift {s['han_contact_shift_cm']:+.2f} cm) + horizontal "
                                       f"{s['horizontal_shift_cm'][:2]} cm; frame-0 contact {s['frame0_min_sdf_after_cm']} cm; path clearance {s['path_min_sdf_cm']} cm")
            r, h, g, st = p["rest"], p["human"], p["g1"], p["settle"]
            runs = [x for x in r["runs"] if x["near_support"]] if r else []
            row["phys_rest"] = (", ".join(f"{x['run'][0]}-{x['run'][1]}:{x['median_sdf_cm']:+.2f}cm/cov{x['footprint_cover']:.2f}" for x in runs)
                                if r else "missing")
            row["phys_human"] = f"{h['violating_frames']} viol frames (min sdf {h['min_sdf_cm']} cm)" if h else "missing"
            row["phys_g1"] = (f"{g['violating_frames']} viol frames (min body {g['min_body_clear_cm']} / hand {g['min_hand_clear_cm']} cm)"
                              if g else "missing")
            row["phys_settle"] = (", ".join(f"{x['run'][0]}-{x['run'][1]}:{'ok' if x['with_support']['ok'] else 'FAIL'}"
                                            f"({x['with_support']['final_dp_cm']:.1f}cm,{x['with_support']['final_rot_deg']:.1f}deg)"
                                            for x in st["tests"]) + f"; control moves {st['anchor_control_moves']}") if st else "missing"
            row["flags"] += p["flags"]
            row["scene_support"] = True
        else:
            row.update(support="floor-only", support_geometry="-", support_pose="-", phys_rest="-", phys_human="-",
                       phys_g1="-", phys_settle="-", scene_support=False)
        rows.append(row)
    (A / "build/prep_table.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    cols = ["ep", "object", "frames", "support", "rcheck_all_ok", "robot_max_diff_vs_opus", "hand_contact_frames_LR_f0",
            "hand_contact_frames_LR_opus", "object_ground_lift", "track_change_vs_opus", "support_geometry", "support_pose",
            "phys_rest", "phys_human", "phys_g1", "phys_settle", "flags"]
    with (A / "build/prep_table.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
