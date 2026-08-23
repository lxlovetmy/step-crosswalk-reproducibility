# Direction-Specific Crosswalk Resource: Safe Use

## What this resource does

The individual crosswalk conditionally re-expresses a participant's **multi-day mean daily steps** from one named algorithm scale to another. It covers 42 independently fitted directions across seven algorithms. `A -> B` and `B -> A` are different resources and are not algebraic inverses.

This resource does **not** estimate true steps, establish algorithm-neutral health thresholds, validate a device outside the NHANES wrist-signal resource, or align population prevalence. The separate sample-level alignment lookup serves a different sample-marginal purpose and must not be used for individual conversion.

## Files

- `crosswalk_exact_knots.csv`: exact fitted isotonic thresholds for machine-readable use.
- `crosswalk_direction_metadata.csv`: one row per direction with labels, units, support, error evidence, version, and hashes.
- `analysis/crosswalk_converter.py`: strict converter using the exact thresholds.

No file contains participant identifiers, participant-level predictions, person-days, minutes, or bootstrap-replicate rows.

## Required inputs

1. Use an algorithm code listed in `crosswalk_direction_metadata.csv`.
2. Specify source and target explicitly.
3. Supply participant-level multi-day mean daily steps in `steps/day`.
4. Keep the value within that direction's `released_source_p05_steps` to `released_source_p95_steps` interval.

Do not supply a single day, raw acceleration, cadence, a population percentile, or a health threshold and interpret the result as an individual conversion.

## Strict converter

Example from the package root:

```bash
python analysis/crosswalk_converter.py \
  --source acti \
  --target oak \
  --value 8000
```

Repeat `--value` for multiple values. Add `--round-whole-step` only when an integer output is required; rounding occurs after interpolation.

The converter returns one of four statuses:

- `IN_RANGE`: exact-threshold interpolation was performed.
- `OUT_OF_RANGE_BELOW`: no formal conversion was assigned.
- `OUT_OF_RANGE_ABOVE`: no formal conversion was assigned.
- `INVALID_NONFINITE_INPUT`: no conversion was assigned.

It never silently clips or extrapolates. An unsupported or reversed direction produces an error rather than guessing.

## Exact thresholds and the direction index

The exact thresholds reproduce the fitted scikit-learn isotonic mapping by linear interpolation and are the sole formal conversion parameters. Supplemental Table S2 and `crosswalk_direction_metadata.csv` index the 42 directions and state the source P05–P95 use range, exact-knot count, and direction-level E/H evidence. The index is documentation, not a second approximation of the fitted function.

## What to report when using a direction

At minimum, report:

- source and target algorithms;
- resource version;
- analysis level (multi-day mean daily steps);
- released source range used;
- that the exact-threshold resource and strict converter were used;
- target-scale OOF MAE or E from the direction metadata;
- classification consequences at an intended fixed numerical value when relevant.

Use H and common-support results as support-sensitive diagnostics, not as true-step accuracy scores or universal permission grades.
