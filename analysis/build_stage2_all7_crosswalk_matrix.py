#!/usr/bin/env python3
"""Stage 2a: extend the locked MVP crosswalk machinery to all 7 step algorithms.

Exposure-only (no HbA1c/waist linkage, no models). On the locked D9-valid adult
cohort it builds:
- per-algorithm daily steps + peak-30 cadence distributions (7 algorithms),
- pairwise agreement matrix (21 unordered pairs): Spearman/Pearson, abs diff, Bland-Altman,
- pairwise crosswalk translatability matrix (42 ordered pairs, daily steps):
  isotonic 5-fold out-of-fold residual (MAE/RMSE, normalized by target IQR),
- pairwise fixed-threshold reclassification (42 ordered pairs x 7000/8000/10000,
  daily steps): raw and post-crosswalk discordance + kappa.

Methodology is fully reused from the locked SAP v1.3 / D8 / D9 / D10 decisions;
no new method decision is made here. Outputs are aggregate tables and a run log.
No SEQN/person-day/minute-level output is written.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from build_mvp_exposure_qc import (
    TROIANO_FILE,
    collect_d9_valid_days,
    collect_wear_masks,
    pearson_r,
    process_step_algorithm,
    read_adult_seqns_from_demo,
    read_step_index,
    spearman_r,
    write_csv,
)
from build_mvp_crosswalk_stability import (
    FIXED_STEP_THRESHOLDS,
    FOLDS,
    RANDOM_SEED,
    fit_crosswalk,
    kappa,
)
from gate0_d9_wear_audit import locate_file, pct, read_day_minmax, summary_stats

ALL_STEP_FILES = {
    "acti": "nhanes_1440_actisteps.csv.xz",
    "adept": "nhanes_1440_adeptsteps.csv.xz",
    "oak": "nhanes_1440_oaksteps.csv.xz",
    "scrf": "nhanes_1440_scrfsteps.csv.xz",
    "scssl": "nhanes_1440_scsslsteps.csv.xz",
    "vs": "nhanes_1440_vssteps.csv.xz",
    "vsrev": "nhanes_1440_vsrevsteps.csv.xz",
}
ALGORITHMS = list(ALL_STEP_FILES.keys())
HEADLINE_METHOD = "isotonic"
DAILY_KEY = "daily_steps_whole_day_sum_mean"
CADENCE_KEY = "peak30_cadence_mean_secondary"
DEGENERATE_POS_RATE = 0.02  # target positive rate below this (or above 1-this) => fixed threshold degenerate


def log(msg: str) -> None:
    print(f"[stage2a {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def aligned_array(metrics: dict[str, dict[str, float]], subjects: list[str], key: str) -> np.ndarray:
    return np.asarray(
        [float(metrics.get(seqn, {}).get(key, math.nan)) for seqn in subjects],
        dtype=float,
    )


def oof_isotonic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.full(len(x), np.nan, dtype=float)
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in splitter.split(x):
        fitted = fit_crosswalk(HEADLINE_METHOD, x[train_idx], y[train_idx])
        pred[test_idx] = fitted.predict(x[test_idx])
    return pred


def distribution_rows(
    daily_by_alg: dict[str, np.ndarray],
    cadence_by_alg: dict[str, np.ndarray],
    n_subjects: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for exposure, arrays in (("daily_steps", daily_by_alg), ("peak30_cadence", cadence_by_alg)):
        for alg in ALGORITHMS:
            values = arrays[alg]
            clean = [v for v in values.tolist() if not math.isnan(v)]
            stats = summary_stats(clean)
            rows.append(
                {
                    "exposure": exposure,
                    "algorithm": alg,
                    "n_subjects_denominator": n_subjects,
                    "n_nonmissing": len(clean),
                    "pct_missing": pct(n_subjects - len(clean), n_subjects),
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "p05": stats["p05"],
                    "p25": stats["p25"],
                    "median": stats["p50"],
                    "p75": stats["p75"],
                    "p95": stats["p95"],
                    "iqr": float(stats["p75"]) - float(stats["p25"]),
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )
    return rows


def agreement_rows(
    daily_by_alg: dict[str, np.ndarray],
    cadence_by_alg: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for exposure, arrays in (("daily_steps", daily_by_alg), ("peak30_cadence", cadence_by_alg)):
        for a, b in itertools.combinations(ALGORITHMS, 2):
            xa = arrays[a]
            xb = arrays[b]
            mask = ~np.isnan(xa) & ~np.isnan(xb)
            xs = xa[mask].tolist()
            ys = xb[mask].tolist()
            if len(xs) < 2:
                continue
            diffs = [x - y for x, y in zip(xs, ys)]
            abs_diffs = [abs(d) for d in diffs]
            diff_stats = summary_stats(diffs)
            absd_stats = summary_stats(abs_diffs)
            mean_diff = float(diff_stats["mean"])
            sd_diff = float(diff_stats["sd"])
            rows.append(
                {
                    "exposure": exposure,
                    "algorithm_a": a,
                    "algorithm_b": b,
                    "n_pair": len(xs),
                    "pearson_r": pearson_r(xs, ys),
                    "spearman_r": spearman_r(xs, ys),
                    "mean_a_minus_b": mean_diff,
                    "median_a_minus_b": diff_stats["p50"],
                    "median_abs_diff": absd_stats["p50"],
                    "p95_abs_diff": absd_stats["p95"],
                    "bland_altman_lower_loa": mean_diff - 1.96 * sd_diff if not math.isnan(sd_diff) else math.nan,
                    "bland_altman_upper_loa": mean_diff + 1.96 * sd_diff if not math.isnan(sd_diff) else math.nan,
                    "note": "Unordered pairwise agreement on subjects with both exposures non-missing.",
                }
            )
    return rows


def crosswalk_residual_rows(
    daily_by_alg: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    oof_cache: dict[tuple[str, str], np.ndarray] = {}
    for src, tgt in itertools.permutations(ALGORITHMS, 2):
        x = daily_by_alg[src]
        y = daily_by_alg[tgt]
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_use = x[mask]
        y_use = y[mask]
        pred = oof_isotonic(x_use, y_use)
        oof_cache[(src, tgt)] = pred  # aligned to mask order; only used internally for reclass below with same mask
        residual = pred - y_use
        abs_err = np.abs(residual)
        tgt_stats = summary_stats(y_use.tolist())
        tgt_iqr = float(tgt_stats["p75"]) - float(tgt_stats["p25"])
        mae = float(np.mean(abs_err)) if len(abs_err) else math.nan
        rmse = math.sqrt(float(np.mean(residual ** 2))) if len(residual) else math.nan
        rows.append(
            {
                "exposure": "daily_steps",
                "method": HEADLINE_METHOD,
                "source_algorithm": src,
                "target_algorithm": tgt,
                "n": int(len(y_use)),
                "bias_pred_minus_target": float(np.mean(residual)) if len(residual) else math.nan,
                "mae": mae,
                "rmse": rmse,
                "abs_error_p95": float(np.percentile(abs_err, 95)) if len(abs_err) else math.nan,
                "target_iqr": tgt_iqr,
                "mae_per_target_iqr": mae / tgt_iqr if tgt_iqr and tgt_iqr > 0 else math.nan,
                "note": "Isotonic source->target crosswalk, 5-fold out-of-fold; residual on target scale.",
            }
        )
    return rows, oof_cache


def reclassification_rows(daily_by_alg: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for src, tgt in itertools.permutations(ALGORITHMS, 2):
        x = daily_by_alg[src]
        y = daily_by_alg[tgt]
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_use = x[mask]
        y_use = y[mask]
        pred = oof_isotonic(x_use, y_use)
        n = int(len(y_use))
        for threshold in FIXED_STEP_THRESHOLDS:
            tgt_class = y_use >= threshold
            tgt_pos_rate = float(np.mean(tgt_class)) if n else math.nan
            degenerate = (tgt_pos_rate < DEGENERATE_POS_RATE) or (tgt_pos_rate > 1 - DEGENERATE_POS_RATE)
            for stage, classed in (
                ("before_crosswalk_raw_same_numeric", x_use >= threshold),
                ("after_crosswalk_oof_isotonic", pred >= threshold),
            ):
                discordant = classed != tgt_class
                rows.append(
                    {
                        "exposure": "daily_steps",
                        "source_algorithm": src,
                        "target_algorithm": tgt,
                        "threshold_target_steps": threshold,
                        "stage": stage,
                        "n": n,
                        "source_or_converted_positive_pct": pct(int(np.sum(classed)), n),
                        "target_positive_pct": tgt_pos_rate,
                        "discordant_pct": pct(int(np.sum(discordant)), n),
                        "kappa": kappa(classed, tgt_class),
                        "target_threshold_degenerate": degenerate,
                        "note": "Degenerate=target positive rate <2% or >98% (e.g., adept as target at mainstream thresholds); kappa unreliable.",
                    }
                )
    return rows


def write_summary(
    path: Path,
    denom_rows: list[dict[str, object]],
    dist_rows: list[dict[str, object]],
    agree_rows: list[dict[str, object]],
    residual_rows: list[dict[str, object]],
    reclass_rows: list[dict[str, object]],
    n_subjects: int,
    n_valid_days: int,
) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    denom = {row["metric"]: row["value"] for row in denom_rows}
    daily_dist = {row["algorithm"]: row for row in dist_rows if row["exposure"] == "daily_steps"}
    lines = [
        "# Stage 2a All-7 Algorithm Crosswalk Matrix Summary",
        "",
        f"- Generated: {generated}",
        "- Scope: exposure-only extension of the locked MVP crosswalk machinery to all 7 step algorithms.",
        f"- Denominator: {n_subjects} D9-valid adults (>=20), {n_valid_days} valid person-days, rebuilt in memory.",
        "- D9 main rule, daily-steps and cadence definitions, isotonic headline crosswalk, 5-fold OOF: all reused from SAP v1.3 (D8/D9/D10).",
        "- No HbA1c/waist linkage, no models, no SEQN/person-day/minute output, SAP unchanged.",
        "",
        "## Cohort Check (7-algorithm intersection should match MVP 8,646)",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for metric in [
        "per_algorithm_step_person_days",
        "per_algorithm_step_subjects",
        "seven_way_intersection_person_days",
        "seven_way_intersection_subjects",
        "troiano_intersect_seven_way_person_days",
        "d9_valid_adult_subjects_final",
        "d9_valid_person_days_final",
    ]:
        lines.append(f"| {metric} | {denom.get(metric, 'NA')} |")
    lines.extend(
        [
            "",
            "## Daily Steps Distribution by Algorithm",
            "",
            "| algorithm | mean | median | IQR | p05 | p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for alg in ALGORITHMS:
        r = daily_dist[alg]
        lines.append(
            "| {a} | {mean} | {med} | {iqr} | {p05} | {p95} |".format(
                a=alg,
                mean=f"{float(r['mean']):.1f}",
                med=f"{float(r['median']):.1f}",
                iqr=f"{float(r['iqr']):.1f}",
                p05=f"{float(r['p05']):.1f}",
                p95=f"{float(r['p95']):.1f}",
            )
        )
    # Spearman matrix (daily)
    lines.extend(["", "## Daily Steps Pairwise Spearman (unordered)", "", "| pair | Spearman | median abs diff |", "|---|---:|---:|"])
    for row in agree_rows:
        if row["exposure"] != "daily_steps":
            continue
        lines.append(
            "| {a} vs {b} | {s} | {d} |".format(
                a=row["algorithm_a"],
                b=row["algorithm_b"],
                s=f"{float(row['spearman_r']):.3f}",
                d=f"{float(row['median_abs_diff']):.1f}",
            )
        )
    # Crosswalk residual: highlight best/worst by mae_per_target_iqr
    lines.extend(
        [
            "",
            "## Daily Steps Crosswalk Translatability (isotonic OOF, ordered source->target)",
            "",
            "Lower MAE/target-IQR = more translatable. Full 42-pair table in CSV.",
            "",
            "| source -> target | MAE (target steps) | MAE / target IQR |",
            "|---|---:|---:|",
        ]
    )
    ranked = sorted(residual_rows, key=lambda r: float(r["mae_per_target_iqr"]))
    show = ranked[:5] + ranked[-5:]
    for row in show:
        lines.append(
            "| {s} -> {t} | {mae} | {norm} |".format(
                s=row["source_algorithm"],
                t=row["target_algorithm"],
                mae=f"{float(row['mae']):.1f}",
                norm=f"{float(row['mae_per_target_iqr']):.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Pairwise reclassification (raw vs post-crosswalk, 7000/8000/10000) is in `stage2_all7_reclassification_kappa.csv`; rows with `target_threshold_degenerate=True` (e.g. adept as target) have unreliable kappa.",
            "- Subgroup (age/sex/BMI) curve-split stability was established for adept->oak in the MVP crosswalk round; a focused all-pair subgroup stability pass can follow if needed.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_tables = out_dir / "tables"
    out_logs = out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    physio_dir = raw_dir / "physionet_v1.0.1"
    troiano_path = locate_file(physio_dir, TROIANO_FILE)
    step_paths = {alg: locate_file(physio_dir, fname) for alg, fname in ALL_STEP_FILES.items()}

    log("reading adult SEQNs from DEMO (R)")
    adult_seqns = read_adult_seqns_from_demo(raw_dir)

    log("reading 7 step indices")
    step_indices = {alg: read_step_index(path) for alg, path in step_paths.items()}
    per_alg_pd = {alg: len(idx) for alg, idx in step_indices.items()}
    per_alg_subj = {alg: len({s for s, _d in idx}) for alg, idx in step_indices.items()}
    seven_way = set.intersection(*step_indices.values())
    log(f"7-way intersection person-days={len(seven_way)} subjects={len({s for s,_ in seven_way})}")

    minmax_by_seqn = read_day_minmax(troiano_path)
    _valid_by_subj, final_subjects, final_valid_days, _align = collect_d9_valid_days(
        troiano_path=troiano_path,
        step_common_keys=seven_way,
        minmax_by_seqn=minmax_by_seqn,
        adult_seqns=adult_seqns,
    )
    log(f"D9-valid adults final={len(final_subjects)} valid person-days={len(final_valid_days)}")

    wear_masks = collect_wear_masks(troiano_path, final_valid_days)

    metrics_by_algorithm: dict[str, dict[str, dict[str, float]]] = {}
    for alg, path in step_paths.items():
        log(f"processing exposures: {alg}")
        metrics, _agg = process_step_algorithm(alg, path, wear_masks)
        metrics_by_algorithm[alg] = metrics

    subjects_sorted = sorted(final_subjects)
    daily_by_alg = {alg: aligned_array(metrics_by_algorithm[alg], subjects_sorted, DAILY_KEY) for alg in ALGORITHMS}
    cadence_by_alg = {alg: aligned_array(metrics_by_algorithm[alg], subjects_sorted, CADENCE_KEY) for alg in ALGORITHMS}

    # consistency check: all step files share the same index
    distinct_pd = set(per_alg_pd.values())
    distinct_subj = set(per_alg_subj.values())
    denom_rows = [
        {"metric": "per_algorithm_step_person_days", "value": sorted(distinct_pd), "note": "Each of 7 step files; should be identical."},
        {"metric": "per_algorithm_step_subjects", "value": sorted(distinct_subj), "note": "Each of 7 step files; should be identical."},
        {"metric": "seven_way_intersection_person_days", "value": len(seven_way), "note": "Person-days present in all 7 step files."},
        {"metric": "seven_way_intersection_subjects", "value": len({s for s, _d in seven_way}), "note": "Subjects present in all 7 step files."},
        {"metric": "troiano_intersect_seven_way_person_days", "value": len(final_valid_days) , "note": "Final D9-valid person-days used."},
        {"metric": "d9_valid_adult_subjects_final", "value": len(final_subjects), "note": "Analysis denominator subjects."},
        {"metric": "d9_valid_person_days_final", "value": len(final_valid_days), "note": "Analysis denominator valid person-days."},
    ]

    log("building distribution rows")
    dist_rows = distribution_rows(daily_by_alg, cadence_by_alg, len(subjects_sorted))
    log("building agreement matrix (21 pairs x 2 exposures)")
    agree_rows = agreement_rows(daily_by_alg, cadence_by_alg)
    log("building crosswalk residual matrix (42 ordered pairs, isotonic OOF)")
    residual_rows, _oof = crosswalk_residual_rows(daily_by_alg)
    log("building reclassification matrix (42 ordered pairs x 3 thresholds)")
    reclass_rows = reclassification_rows(daily_by_alg)

    write_csv(
        out_tables / "stage2_all7_cohort_check.csv",
        [{"metric": r["metric"], "value": r["value"], "note": r["note"]} for r in denom_rows],
        ["metric", "value", "note"],
    )
    write_csv(
        out_tables / "stage2_all7_distribution.csv",
        dist_rows,
        ["exposure", "algorithm", "n_subjects_denominator", "n_nonmissing", "pct_missing",
         "mean", "sd", "p05", "p25", "median", "p75", "p95", "iqr", "min", "max"],
    )
    write_csv(
        out_tables / "stage2_all7_pairwise_agreement.csv",
        agree_rows,
        ["exposure", "algorithm_a", "algorithm_b", "n_pair", "pearson_r", "spearman_r",
         "mean_a_minus_b", "median_a_minus_b", "median_abs_diff", "p95_abs_diff",
         "bland_altman_lower_loa", "bland_altman_upper_loa", "note"],
    )
    write_csv(
        out_tables / "stage2_all7_crosswalk_residual_matrix.csv",
        residual_rows,
        ["exposure", "method", "source_algorithm", "target_algorithm", "n",
         "bias_pred_minus_target", "mae", "rmse", "abs_error_p95", "target_iqr",
         "mae_per_target_iqr", "note"],
    )
    write_csv(
        out_tables / "stage2_all7_reclassification_kappa.csv",
        reclass_rows,
        ["exposure", "source_algorithm", "target_algorithm", "threshold_target_steps",
         "stage", "n", "source_or_converted_positive_pct", "target_positive_pct",
         "discordant_pct", "kappa", "target_threshold_degenerate", "note"],
    )
    write_summary(
        out_tables / "STAGE2_ALL7_CROSSWALK_MATRIX_SUMMARY.md",
        denom_rows, dist_rows, agree_rows, residual_rows, reclass_rows,
        len(subjects_sorted), len(final_valid_days),
    )

    finished = datetime.now(timezone.utc)
    run_log = {
        "script": "scripts/python/build_stage2_all7_crosswalk_matrix.py",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "algorithms": ALGORITHMS,
        "headline_method": HEADLINE_METHOD,
        "folds": FOLDS,
        "random_seed": RANDOM_SEED,
        "denominator_subjects": len(final_subjects),
        "denominator_valid_days": len(final_valid_days),
        "seven_way_intersection_person_days": len(seven_way),
        "outputs": [
            "outputs/tables/stage2_all7_cohort_check.csv",
            "outputs/tables/stage2_all7_distribution.csv",
            "outputs/tables/stage2_all7_pairwise_agreement.csv",
            "outputs/tables/stage2_all7_crosswalk_residual_matrix.csv",
            "outputs/tables/stage2_all7_reclassification_kappa.csv",
            "outputs/tables/STAGE2_ALL7_CROSSWALK_MATRIX_SUMMARY.md",
            "outputs/logs/stage2_all7_crosswalk_matrix_run.json",
        ],
        "guardrails": [
            "in-memory exposure reconstruction only",
            "aggregate tables only",
            "no SEQN/person-day/minute-level outputs",
            "no HbA1c/waist linkage",
            "no exposure-outcome models",
            "no new SAP method lock; reuses SAP v1.3 D8/D9/D10",
        ],
    }
    with (out_logs / "stage2_all7_crosswalk_matrix_run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_log, handle, ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main()
