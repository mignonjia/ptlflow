# World Model Evaluation Changes

## Summary

These staged changes make synthetic-flow evaluation usable for checkpoint selection, not just one-off inspection.

## Changes

`eval_flow_divergence.py`

- Added a synthetic auto-sweep mode for evaluating many `validation_step_*` videos across training steps and validation indices.
- It now reads `action_path` from `validation_json`, discovers matching generated videos under `video_root`, runs evaluation for each `(train_step, validation_idx)` pair, and writes outputs under `output_dir/idx_{idx}/step_{step}/`.
- It now aggregates all run summaries into:
  `all_summaries.{json,csv}`, `step_average_summaries.{json,csv}`, `score_change_trend.csv`, `metric_trends_3x3.png`, `best_steps_by_metric.json`, and `best_step.json`.
- Mean-flow angle/cosine metrics now handle tiny vectors explicitly:
  both tiny means match, one tiny and one non-tiny counts as mismatch.
- When synthetic evaluation trims usable frames, it now trims action arrays too.
- Default `--ckpt` now points to a specific local PTLFlow checkpoint instead of a generic alias.

Why:
- The old script was single-run oriented.
- These changes make checkpoint comparison reproducible and easier to summarize.
- The metric and trimming fixes reduce misleading results from near-zero motion and sequence-length mismatch.

`plot_metric_trends.py`

- Added a small plotting utility for `step_average_summaries.csv`.
- It plots the 9 main metrics in a 3x3 grid and marks the best point for each metric.

Why:
- This gives a quick visual summary of how metrics change over training steps without rerunning evaluation.

`synthetic_flow.py`

- Synthetic flow generation now caps the number of flows to `len(frames) - 1` when frames are provided.

Why:
- This keeps synthetic flow length aligned with available frame pairs and avoids action/video length mismatch.

`.gitignore`

- Added `*.ckpt`.

Why:
- Local checkpoint files are large machine-specific artifacts and should not appear as git changes.

## Net Effect

The workflow now supports:
- multi-checkpoint synthetic evaluation,
- step-level aggregation across validation samples,
- automatic best-step selection,
- and more stable behavior for tiny-motion and length-mismatch cases.
