#!/usr/bin/env python3
"""Build submission-table data layers and Figures 1-4/S1-S4."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_MPL = tempfile.TemporaryDirectory(prefix="step_crosswalk_mpl_")
os.environ.setdefault("MPLCONFIGDIR", _MPL.name)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from build_mvp_exposure_qc import spearman_r
from build_stage2c_all7_subgroup_stability import build_daily_cohort
from release_common import ALGORITHMS, ALGORITHM_LABELS, DIRECTED_PAIRS


COLORS = ["#dcae27", "#d65f9e", "#2f78b7", "#3d9c5b", "#efc94c", "#68b0cc", "#e77c32"]
SELECTED = [
    ("oak", "scrf", "Smaller OOF-error example"),
    ("oak", "vsrev", "Smaller OOF-error example"),
    ("scrf", "oak", "Smaller OOF-error example"),
    ("acti", "oak", "P05-P95 subgroup-diagnostic example"),
    ("vs", "oak", "P05-P95 subgroup-diagnostic example"),
    ("vsrev", "oak", "P05-P95 subgroup-diagnostic example"),
    ("adept", "oak", "Larger OOF-error example"),
    ("oak", "adept", "Larger OOF-error example"),
    ("scssl", "oak", "Larger OOF-error example"),
]


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: list[float]) -> list[float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x, w = values[mask], weights[mask]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cumulative = np.cumsum(w)
    return [float(x[np.searchsorted(cumulative, p * cumulative[-1], side="left")]) for p in probabilities]


def read_table1_frame(raw_dir: Path, cohort) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    for cycle, suffix in (("nhanes_2011_2012", "G"), ("nhanes_2013_2014", "H")):
        demo = pd.read_sas(raw_dir / cycle / f"DEMO_{suffix}.XPT", format="xport", encoding="latin1")
        bmx = pd.read_sas(raw_dir / cycle / f"BMX_{suffix}.XPT", format="xport", encoding="latin1")
        keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "WTMEC2YR"]
        demo = demo[[column for column in keep if column in demo]].copy()
        frames.append(demo.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left"))
    frame = pd.concat(frames, ignore_index=True)
    frame["WTMEC2YR"] = frame["WTMEC2YR"].mask(frame["WTMEC2YR"].abs() < 1e-70)
    positive_mec_subjects = {
        str(int(float(value)))
        for value in frame.loc[frame["WTMEC2YR"].gt(0), "SEQN"]
    }
    flow_audit: dict[str, object] = {
        "positive_mec_weight_subjects": len(positive_mec_subjects),
        **cohort.sample_flow_counts,
        "seven_way_not_in_positive_mec_weight_n": len(
            cohort.seven_way_subject_ids - positive_mec_subjects
        ),
    }
    expected_flow = {
        "positive_mec_weight_subjects": 19151,
        "seven_way_subjects": 14693,
        "seven_way_person_days": 130186,
        "troiano_intersection_subjects": 14689,
        "troiano_intersection_person_days": 130179,
        "valid_wear_subjects_before_adult_filter": 13137,
        "final_adult_subjects": 8646,
        "final_valid_person_days": 57080,
        "seven_way_not_in_positive_mec_weight_n": 0,
    }
    if any(flow_audit[key] != expected for key, expected in expected_flow.items()):
        raise RuntimeError(
            f"Sample-flow assertion failed: observed={flow_audit}, expected={expected_flow}"
        )
    flow_audit["assertion"] = "PASS"
    frame["seqn"] = frame["SEQN"].map(lambda value: str(int(float(value))))
    frame = frame.set_index("seqn").loc[[str(value) for value in cohort.subject_ids]].reset_index()
    frame["weight"] = frame["WTMEC2YR"] / 2.0
    frame["oak_daily"] = np.asarray(cohort.daily_by_alg["oak"], dtype=float)
    race = frame["RIDRETH3"].where(frame["RIDRETH3"].notna(), frame["RIDRETH1"])
    frame["sex"] = frame["RIAGENDR"].map({1: "Male", 2: "Female"})
    frame["race"] = race.map({1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/multiracial"})
    frame["education"] = frame["DMDEDUC2"].map({1: "<9th grade", 2: "9th-11th grade", 3: "High school/GED", 4: "Some college/AA", 5: "College graduate or above"})
    frame["age_group"] = pd.cut(frame["RIDAGEYR"], [19, 39, 59, np.inf], labels=["20-39", "40-59", "60+"]).astype(object)
    if len(frame) != 8646:
        raise RuntimeError(f"Table 1 cohort alignment failed: {len(frame)}")
    return frame, flow_audit


def table1_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    w = frame["weight"].to_numpy(float)

    def continuous(variable, label, column, kind):
        values = frame[column].to_numpy(float)
        mask = np.isfinite(values) & np.isfinite(w) & (w > 0)
        if kind == "mean_sd":
            mean = float(np.average(values[mask], weights=w[mask]))
            sd = float(np.sqrt(np.average((values[mask] - mean) ** 2, weights=w[mask])))
            estimate = f"{mean:.1f} ({sd:.1f})"
        else:
            q25, q50, q75 = weighted_quantile(values, w, [0.25, 0.5, 0.75])
            if variable == "oak_daily_steps":
                estimate = f"{q50:,.0f} ({q25:,.0f}–{q75:,.0f})"
            else:
                estimate = f"{q50:.1f} ({q25:.1f}–{q75:.1f})"
        rows.append({"characteristic": label, "available_n": int(mask.sum()), "missing_n": int(len(frame)-mask.sum()), "survey_weighted_estimate": estimate, "row_type": "data"})

    def categorical(variable, label, column, levels, show_header=True):
        valid = frame[column].notna() & frame["weight"].notna() & frame["weight"].gt(0)
        denominator = float(frame.loc[valid, "weight"].sum())
        if show_header:
            rows.append({"characteristic": label, "available_n": int(valid.sum()), "missing_n": int(len(frame)-valid.sum()), "survey_weighted_estimate": "", "row_type": "section"})
        for level in levels:
            numerator = float(frame.loc[valid & frame[column].eq(level), "weight"].sum())
            rows.append({"characteristic": f"  {level}", "available_n": "", "missing_n": "", "survey_weighted_estimate": f"{100*numerator/denominator:.1f}", "row_type": "data"})

    continuous("age_years", "Age, y — mean (SD)", "RIDAGEYR", "mean_sd")
    categorical("age_group", "Age group, %", "age_group", ["20-39", "40-59", "60+"])
    valid_sex = frame["sex"].notna() & frame["weight"].notna() & frame["weight"].gt(0)
    female = 100 * frame.loc[valid_sex & frame["sex"].eq("Female"), "weight"].sum() / frame.loc[valid_sex, "weight"].sum()
    rows.append({"characteristic": "Female, %", "available_n": int(valid_sex.sum()), "missing_n": int(len(frame)-valid_sex.sum()), "survey_weighted_estimate": f"{female:.1f}", "row_type": "data"})
    categorical("race", "Race/ethnicity, %", "race", ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other/multiracial"])
    categorical("education", "Education, %", "education", ["<9th grade", "9th-11th grade", "High school/GED", "Some college/AA", "College graduate or above"])
    continuous("pir", "Family income-to-poverty ratio — median (IQR)", "INDFMPIR", "median_iqr")
    continuous("bmi", "Body mass index, kg/m² — mean (SD)", "BMXBMI", "mean_sd")
    continuous("oak_daily_steps", "Daily steps (Oak illustrative scale), steps/d — median (IQR)", "oak_daily", "median_iqr")
    result = pd.DataFrame(rows)
    if len(result) != 22:
        raise RuntimeError(f"Main Table 1 must contain 22 display rows, got {len(result)}")
    return result


def table2(distribution: pd.DataFrame) -> pd.DataFrame:
    frame = distribution.loc[distribution["exposure"].eq("daily_steps")].copy()
    frame["Algorithm"] = frame["algorithm"].map(ALGORITHM_LABELS)
    frame = frame.rename(columns={"n_nonmissing": "n", "median": "Median", "p25": "P25", "p75": "P75", "iqr": "IQR", "p05": "P05", "p95": "P95", "min": "Minimum", "max": "Maximum"})
    return frame[["Algorithm", "n", "Median", "P25", "P75", "IQR", "P05", "P95", "Minimum", "Maximum"]]


def table3(pair: pd.DataFrame, common: pd.DataFrame, cross: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, target, role in SELECTED:
        p = pair.loc[pair["source_algorithm"].eq(source) & pair["target_algorithm"].eq(target)].iloc[0]
        c = common.loc[common["source_algorithm"].eq(source) & common["target_algorithm"].eq(target)].iloc[0]
        delta = cross.loc[cross["source_algorithm"].eq(source) & cross["target_algorithm"].eq(target), "transport_penalty_mae_per_target_iqr"].astype(float).max()
        f = fixed.loc[
            fixed["source_algorithm"].eq(source) & fixed["target_algorithm"].eq(target)
            & fixed["threshold_target_steps"].eq(8000.0) & fixed["analysis_weighting"].eq("unweighted_primary")
        ].iloc[0]
        rows.append({
            "Source_to_target": f"{ALGORITHM_LABELS[source]} → {ALGORITHM_LABELS[target]}",
            "Illustrative_role": role,
            "OOF_MAE_over_IQR": float(p["E"]),
            "Complete_sample_P05_P95_H": float(p["H"]),
            "Common_support_max_spread_over_IQR": float(c["common_support_empirical_H"]),
            "Less_favorable_delta_MAE_over_IQR": float(delta),
            "OOF_discordance_at_8000_pct": float(f["discordant_reclassification_pct"]),
            "Cohen_kappa": float(f["cohen_kappa"]),
            "Degenerate_target": bool(f["target_threshold_degenerate"]),
        })
    return pd.DataFrame(rows)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1(fixed: pd.DataFrame, path: Path) -> None:
    data = fixed.loc[fixed["threshold_target_steps"].eq(8000) & fixed["analysis_weighting"].eq("WTMEC4YR_weighted_sensitivity")]
    rates = data.groupby("target_algorithm")["observed_target_attainment_pct"].first().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = [COLORS[ALGORITHMS.index(code)] for code in rates.index]
    ax.barh([ALGORITHM_LABELS[x] for x in rates.index], rates.values, color=colors)
    for y, value in enumerate(rates.values): ax.text(value + 1, y, f"{value:.1f}%", va="center", fontweight="bold")
    ax.set(xlabel="Weighted attainment rate (%)", title="Fixed 8,000-step attainment by algorithm", xlim=(0, 95))
    ax.grid(axis="x", alpha=.3); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def figure2(knots: pd.DataFrame, metadata: pd.DataFrame, path: Path) -> None:
    groups = [SELECTED[:3], SELECTED[3:6], SELECTED[6:]]
    titles = ["Examples with smaller normalized OOF error", "Directions selected for subgroup-diagnostic illustration", "Examples with larger normalized OOF error"]
    fig, axes = plt.subplots(1, 3, figsize=(17.75, 6), sharex=True, sharey=True)
    styles = [("#0072B2", "-"), ("#009E73", "--"), ("#CC79A7", ":")]
    for ax, entries, title in zip(axes, groups, titles):
        ax.plot([0, 22500], [0, 22500], "--", color="gray", label="Identity line")
        for (source, target, _), (color, style) in zip(entries, styles):
            part = knots.loc[
                knots["source_algorithm"].eq(source)
                & knots["target_algorithm"].eq(target)
            ].sort_values("knot_index")
            meta = metadata.loc[
                metadata["source_algorithm"].eq(source)
                & metadata["target_algorithm"].eq(target)
            ].iloc[0]
            low = float(meta["released_source_p05_steps"])
            high = float(meta["released_source_p95_steps"])
            exact_x = part["source_input_steps"].to_numpy(float)
            exact_y = part["predicted_target_steps"].to_numpy(float)
            inside = (exact_x > low) & (exact_x < high)
            plot_x = np.concatenate(([low], exact_x[inside], [high]))
            plot_y = np.concatenate((
                [float(np.interp(low, exact_x, exact_y))],
                exact_y[inside],
                [float(np.interp(high, exact_x, exact_y))],
            ))
            ax.plot(plot_x, plot_y, color=color, linestyle=style, linewidth=2.5, label=f"{ALGORITHM_LABELS[source]} → {ALGORITHM_LABELS[target]}")
        ax.set(title=title, xlabel="Source input steps/day", xlim=(0,22500), ylim=(0,22500)); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Mapped target steps/day")
    fig.suptitle("Selected direction-specific source-to-target step crosswalks", fontsize=17)
    fig.text(.5, .01, "Examples are descriptive; curve colors do not define conversion categories.", ha="center", color="dimgray", fontsize=9)
    save_figure(fig, path)


def figure3(pair: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.2))
    fig.subplots_adjust(left=.08, right=.97, bottom=.25, top=.80, wspace=.50)
    labels = [ALGORITHM_LABELS[x] for x in ALGORITHMS]
    for ax, column, title, vmax in zip(axes, ["E", "H"], ["A  Primary individual error: OOF MAE / target IQR (E)", "B  Complete-sample P05–P95 subgroup diagnostic (H)"], [.45, .60]):
        matrix = np.full((7,7), np.nan)
        for _, row in pair.iterrows(): matrix[ALGORITHMS.index(row.source_algorithm), ALGORITHMS.index(row.target_algorithm)] = float(row[column])
        im = ax.imshow(matrix, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(7), labels, rotation=45, ha="right"); ax.set_yticks(range(7), labels); ax.set(title=title, xlabel="Target algorithm", ylabel="Source algorithm")
        for i in range(7):
            for j in range(7):
                if i != j:
                    value=matrix[i,j]; ax.text(j,i,f"{value:.2f}",ha="center",va="center",color="white" if value < vmax*.58 else "black",fontsize=8)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    fig.suptitle("Direction-specific individual error and subgroup-spread diagnostics", fontsize=16)
    fig.text(.5,.035,"E is primary; H is support-sensitive. Neither defines acceptability.",ha="center",color="dimgray")
    save_figure(fig, path)


def figure4(pair: pd.DataFrame, common: pd.DataFrame, cross: pd.DataFrame, fixed: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 10.5)); gs=fig.add_gridspec(2,2,height_ratios=[1,1],hspace=.34,wspace=.25)
    ax=fig.add_subplot(gs[0,0]); merged=pair[["source_algorithm","target_algorithm","H"]].merge(common[["source_algorithm","target_algorithm","common_support_empirical_H"]],on=["source_algorithm","target_algorithm"])
    ax.scatter(merged.H, merged.common_support_empirical_H, color="#009E9E", s=32); lim=max(merged.H.max(), merged.common_support_empirical_H.max())*1.05; ax.plot([0,lim],[0,lim],"--",color="gray")
    # The manuscript reports direction-level H values to three decimals before
    # describing their displayed change. This makes one small exact decrease
    # (-0.000512) display as unchanged and gives the reported median -0.047.
    quantum = Decimal("0.001")
    displayed_change = [
        Decimal(str(float(common_value))).quantize(quantum, rounding=ROUND_HALF_UP)
        - Decimal(str(float(full_value))).quantize(quantum, rounding=ROUND_HALF_UP)
        for common_value, full_value in zip(
            merged.common_support_empirical_H, merged.H
        )
    ]
    ordered_change = sorted(displayed_change)
    midpoint = len(ordered_change) // 2
    displayed_median = (
        (ordered_change[midpoint - 1] + ordered_change[midpoint]) / Decimal(2)
    ).quantize(quantum, rounding=ROUND_HALF_UP)
    displayed_lower = sum(value < 0 for value in displayed_change)
    ax.text(.03,.96,f"{displayed_lower}/42 below identity; median change = {displayed_median:.3f}",transform=ax.transAxes,va="top")
    ax.set(title="A  Common-support sensitivity",xlabel="Complete-sample P05–P95 H",ylabel="Common-support H"); ax.grid(alpha=.25)
    ax=fig.add_subplot(gs[0,1]); penalties=cross.groupby(["source_algorithm","target_algorithm"])["transport_penalty_mae_per_target_iqr"].max().sort_values().reset_index(drop=True)
    ax.vlines(np.arange(1,len(penalties)+1),0,penalties,color="#7f3c8d",alpha=.45); ax.scatter(np.arange(1,len(penalties)+1),penalties,color="#7f3c8d",s=22); ax.axhline(0,color="gray",lw=1)
    high=penalties.iloc[-1]; ax.text(.72,.93,f"Highest: {high:.3f}",transform=ax.transAxes); ax.set(title="B  Cross-cycle application penalty",xlabel="Directed-pair rank (lower to higher penalty)",ylabel="Less favorable ΔMAE / target IQR"); ax.grid(axis="y",alpha=.25)
    ax=fig.add_subplot(gs[1,:]); dis=fixed.loc[fixed.threshold_target_steps.eq(8000)&fixed.analysis_weighting.eq("unweighted_primary")&~fixed.target_threshold_degenerate.astype(bool),"discordant_reclassification_pct"].astype(float).sort_values().reset_index(drop=True)
    ax.scatter(np.arange(1,len(dis)+1),dis,color="#E66100",s=28); ax.axhline(dis.median(),color="gray",lw=2,label=f"Median = {dis.median():.2f}%"); ax.legend(frameon=False)
    ax.set(title="C  Direction-specific threshold discordance",xlabel="Nondegenerate directed-pair rank (lower to higher discordance)",ylabel="OOF discordance at 8,000 target steps (%)"); ax.grid(axis="y",alpha=.25)
    fig.suptitle("Boundary and use-case checks for direction-specific crosswalks",fontsize=17,fontweight="bold")
    fig.text(.5,.02,"Support, temporal, and threshold checks are reported without acceptability grades.",ha="center",color="dimgray")
    save_figure(fig,path)


def figure_s1(cohort, path: Path) -> None:
    fig, axes=plt.subplots(3,7,figsize=(18,9),sharex=True,sharey=True); axes=axes.ravel()
    for ax,(a,b) in zip(axes,combinations(ALGORITHMS,2)):
        x=np.asarray(cohort.daily_by_alg[a]); y=np.asarray(cohort.daily_by_alg[b]); ax.scatter(x,y,s=1,color="black",alpha=.025,rasterized=True)
        xx=np.array([0,50000]); ax.plot(xx,xx,"--",color="gray"); rho=spearman_r(x.tolist(),y.tolist()); ax.text(.05,.86,f"Spearman ρ={rho:.2f}",transform=ax.transAxes)
        ax.set_title(f"{ALGORITHM_LABELS[a]} vs {ALGORITHM_LABELS[b]}",fontsize=8); ax.set_xlim(0,50000); ax.set_ylim(0,50000); ax.grid(alpha=.15)
    fig.suptitle("Subject-level pairwise relationships across step-count algorithms",fontsize=16); fig.supxlabel("Algorithm A steps/day"); fig.supylabel("Algorithm B steps/day"); save_figure(fig,path)


def figure_s2(cohort,path:Path)->None:
    values=[np.asarray(cohort.daily_by_alg[a],float) for a in ALGORITHMS]; fig,ax=plt.subplots(figsize=(12,7)); parts=ax.violinplot(values,showextrema=False,showmedians=False)
    for body,color in zip(parts["bodies"],COLORS): body.set_facecolor(color); body.set_alpha(.35)
    ax.boxplot(values,widths=.22,showfliers=False,patch_artist=True,boxprops={"facecolor":"white","alpha":.85}); ax.set_xticks(range(1,8),[ALGORITHM_LABELS[a] for a in ALGORITHMS],rotation=30,ha="right"); ax.set(title="Individual daily-step distributions by algorithm",ylabel="Subject-level mean daily steps",ylim=(0,50000)); ax.grid(axis="y",alpha=.25); save_figure(fig,path)


def figure_s3(path:Path, flow:dict[str,object])->None:
    fig,ax=plt.subplots(figsize=(13.5,10)); ax.set_xlim(0,13.5); ax.set_ylim(0,12); ax.axis("off")
    mec=int(flow["positive_mec_weight_subjects"]); seven_n=int(flow["seven_way_subjects"]); seven_days=int(flow["seven_way_person_days"]); troiano_n=int(flow["troiano_intersection_subjects"]); troiano_days=int(flow["troiano_intersection_person_days"]); valid_n=int(flow["valid_wear_subjects_before_adult_filter"]); final_n=int(flow["final_adult_subjects"]); final_days=int(flow["final_valid_person_days"])
    main=[(.4,10.3,8.3,1,f"NHANES 2011–2014 MEC survey-design base with positive examination weights\n(n={mec:,})"),(.4,8.0,8.3,1.1,f"Records shared by all seven released step files\n(n={seven_n:,}; {seven_days:,} participant-days)"),(.4,5.7,8.3,1.1,f"Intersection with the released troianowear series\n(n={troiano_n:,}; {troiano_days:,} participant-days)"),(.4,3.4,8.3,1,f"Met the prespecified valid-wear rule (n={valid_n:,})"),(.4,.9,8.3,1.2,f"Final crosswalk cohort: adults aged ≥20 y with all seven series\n(n={final_n:,}; {final_days:,} valid person-days)")]
    side=[(9.2,8.9,3.8,1.25,f"Excluded\nNot in all seven step files\n(n={mec-seven_n:,})"),(9.2,6.6,3.8,1.25,f"Excluded\nNot in the troianowear intersection\n(n={seven_n-troiano_n:,}; {seven_days-troiano_days:,} participant-days)"),(9.2,4.3,3.8,1.15,f"Excluded\nDid not meet the valid-wear rule\n(n={troiano_n-valid_n:,})"),(9.2,1.9,3.8,1.15,f"Excluded\nAge <20 y\n(n={valid_n-final_n:,})")]
    for x,y,w,h,t in main: ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.4)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=10)
    for x,y,w,h,t in side: ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.4)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9)
    for start,end in [(10.25,9.15),(7.95,6.85),(5.65,4.45),(3.35,2.15)]: ax.add_patch(FancyArrowPatch((4.5,start),(4.5,end),arrowstyle="-|>",mutation_scale=14,color="black"))
    for y in [9.55,7.25,4.95,2.45]: ax.add_patch(FancyArrowPatch((8.7,y),(9.2,y),arrowstyle="-|>",mutation_scale=14,color="black"))
    save_figure(fig,path)


def figure_s4(subgroup_grid:pd.DataFrame,path:Path)->None:
    pairs=[("vs","oak"),("vsrev","oak"),("acti","oak")]; fig,axes=plt.subplots(2,3,figsize=(14,8),sharex="col")
    colors={"20-39":"#56B4E9","40-59":"#E69F00","60+":"#D55E00"}
    for col,(source,target) in enumerate(pairs):
        part=subgroup_grid.loc[subgroup_grid.source_algorithm.eq(source)&subgroup_grid.target_algorithm.eq(target)&subgroup_grid.group_type.eq("age_group")]
        full=part[["source_input","predicted_target_full_sample"]].drop_duplicates().sort_values("source_input"); axes[0,col].plot(full.source_input,full.predicted_target_full_sample,color="black",lw=2.4,label="Full sample")
        for level,color in colors.items():
            g=part.loc[part.group_level.eq(level)].sort_values("source_input"); axes[0,col].plot(g.source_input,g.predicted_target_subgroup,color=color,lw=2,label=level); axes[1,col].plot(g.source_input,g.subgroup_minus_full,color=color,lw=1.7)
        axes[0,col].set_title(f"{ALGORITHM_LABELS[source]} → Oak"); axes[1,col].axhline(0,color="black",ls="--"); axes[0,col].grid(alpha=.25); axes[1,col].grid(alpha=.25)
    axes[0,0].legend(frameon=False); axes[0,0].set_ylabel("Mapped target steps/day"); axes[1,0].set_ylabel("Subgroup − full sample"); fig.supxlabel("Source input steps/day"); fig.suptitle("Complete-sample P05–P95 age-group divergence motivating support-restricted sensitivity",fontsize=16); save_figure(fig,path)


def copy_publication_tables(generated:Path,publication:Path)->None:
    publication.mkdir(parents=True,exist_ok=True)
    for stale in publication.glob("TableS*.csv"):
        stale.unlink()
    index_path = publication/"supplemental_table_file_index.csv"
    if index_path.exists():
        index_path.unlink()
    mapping={
        "TableS1_sample_alignment.csv":"sample_alignment.csv",
        "TableS2_exact_isotonic_knots.csv":"crosswalk_exact_knots.csv",
        "TableS2_direction_metadata.csv":"crosswalk_direction_metadata.csv",
        "TableS3_continuous_E_H.csv":"stage3_translatability_map.csv",
        "TableS4_common_support_detail.csv":"stage6_common_support_subgroup_stability.csv",
        "TableS4_common_support_summary.csv":"stage6_common_support_summary.csv",
        "TableS5_cross_cycle.csv":"stage6_cross_cycle_transport_validation.csv",
        "TableS5_cross_cycle_CI.csv":"stage6_cross_cycle_transport_ci.csv",
        "TableS6_fixed_threshold.csv":"stage6_fixed_threshold_oof_reclassification.csv",
        "TableS6_fixed_threshold_CI.csv":"stage6_fixed_threshold_oof_reclassification_ci.csv",
        "TableS7_bootstrap_uncertainty.csv":"stage4_crosswalk_uncertainty_ci.csv",
        "TableS7_continuous_support_bootstrap.csv":"stage6_continuous_bootstrap.csv",
        "TableS9_release_domain_whole_cohort.csv":"release_domain_whole_cohort.csv",
        "TableS9_release_domain_cross_cycle.csv":"release_domain_cross_cycle.csv",
    }
    for public,source in mapping.items(): shutil.copyfile(generated/"tables"/source,publication/public)
    wear=generated/"wear"
    for source in wear.glob("wear_threshold_*.csv"): shutil.copyfile(source,publication/f"TableS8_{source.name}")
    pd.DataFrame([{"supplemental_table":"S1-S9","file":p.name,"aggregate_only":True} for p in sorted(publication.glob("TableS*.csv"))]).to_csv(publication/"supplemental_table_file_index.csv",index=False)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--raw-dir",type=Path,required=True); parser.add_argument("--generated-dir",type=Path,required=True); args=parser.parse_args()
    generated=args.generated_dir; tables=generated/"tables"; figures=generated/"figures"; publication=generated/"publication_tables"; figures.mkdir(parents=True,exist_ok=True)
    cohort=build_daily_cohort(args.raw_dir)
    distribution=read(tables/"stage2_all7_distribution.csv"); pair=read(tables/"stage3_translatability_map.csv"); knots=read(tables/"crosswalk_exact_knots.csv"); metadata=read(tables/"crosswalk_direction_metadata.csv"); subgroup_grid=read(tables/"stage2c_all7_subgroup_curve_grid_daily_headline.csv"); common=read(tables/"stage6_common_support_summary.csv"); cross=read(tables/"stage6_cross_cycle_transport_validation.csv"); fixed=read(tables/"stage6_fixed_threshold_oof_reclassification.csv")
    table1_frame,flow_audit=read_table1_frame(args.raw_dir,cohort); t1=table1_rows(table1_frame); t2=table2(distribution); t3=table3(pair,common,cross,fixed); publication.mkdir(parents=True,exist_ok=True); t1.to_csv(publication/"main_table1.csv",index=False); t2.to_csv(publication/"main_table2.csv",index=False); t3.to_csv(publication/"main_table3.csv",index=False); copy_publication_tables(generated,publication)
    figure1(fixed,figures/"Figure1.png"); figure2(knots,metadata,figures/"Figure2.png"); figure3(pair,figures/"Figure3.png"); figure4(pair,common,cross,fixed,figures/"Figure4.png"); figure_s1(cohort,figures/"FigureS1.png"); figure_s2(cohort,figures/"FigureS2.png"); figure_s3(figures/"FigureS3.png",flow_audit); figure_s4(subgroup_grid,figures/"FigureS4.png")
    (generated/"logs"/"tables_figures_run.json").write_text(json.dumps({"main_table_rows":[len(t1),len(t2),len(t3)],"supplemental_files":len(list(publication.glob("TableS*.csv"))),"figures":8,"figure4_panels":["A","B","C"],"figure4_common_support_display_rule":"round H and common-support H to 3 decimals before displayed comparison","sample_flow_audit":flow_audit},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
