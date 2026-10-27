# Status: blocked by missing input data

The `/work/inputs` directory was empty when inspected. None of the listed
inputs were available, including `video.mp4`, `reconstruction.npz`, and
`episode.json`. No alternative data sources or network access were used.

Consequently, no image evidence, object geometry, world calibration, frame
count, or baseline poses could be examined. No trajectory improvements can
be justified, and no valid `object_poses_world.npz` can be produced. An empty
or fabricated pose array would not meet the requirement of a pose for every
video frame, so that artifact has deliberately not been created.

`solve.py` implements the specified unchanged-baseline fallback. Once a valid
`reconstruction.npz` is supplied, the requested command copies its
`T_world_mesh` values without pose changes into an `(F,4,4)` float64 array:

```sh
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script checks array shape and finite values and fails explicitly when
the baseline is missing. It uses no scratch dependencies, models, or GPU.
No accuracy claim is possible without the inputs. The attempted run against
the supplied directory failed immediately because `reconstruction.npz`
was missing (under one second; GPU time zero).
