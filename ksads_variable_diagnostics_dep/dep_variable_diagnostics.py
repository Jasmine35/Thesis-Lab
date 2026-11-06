"""
Longitudinal availability diagnostics for an ABCD variable
(default: mh_p_ksads__dep__mdd__pres_dx).

Data layout expected (long format):
    participant_id | session_id | ... | mh_p_ksads__dep__mdd__pres_dx
one row per participant per visit.

Or import and call run_diagnostics(df, var) from a notebook.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID = "participant_id"
SES = "session_id"

# variable I'm performing variable diagnostics on. 
DEFAULT_VAR = "mh_p_ksads__dep__mdd__pres_dx"

# Values treated as real answers. Everything else (blank, NaN, 555, 777, 888, 999,
# "n/a", etc.) is treated as NOT a valid 0/1 answer. Edit after looking at Step 2.
VALID_MAP = {"0": 0, "1": 1, "0.0": 0, "1.0": 1}


def load(path):
    path = Path(path)

    # Read everything as text so codes like "555" or "1" are not silently altered.
    if path.suffix == ".parquet":
        return pd.read_parquet(path).astype({ID: str, SES: str})
    sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""])


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def run_diagnostics(df, var=DEFAULT_VAR, outdir="diagnostics_output"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    df = df[[ID, SES, var]].copy()
    df[var] = df[var].astype("string").str.strip()

    # Session order: ABCD session_ids (e.g. ses-00A, ses-01A, ...) sort correctly as text.
    sessions = sorted(df[SES].dropna().unique())
    df[SES] = pd.Categorical(df[SES], categories=sessions, ordered=True)

    # ---------------------------------------------------------------- Step 1
    header("STEP 1. Structure: one row per participant x session?")
    n_dup = df.duplicated([ID, SES]).sum()
    print(f"Rows: {len(df):,} | Participants: {df[ID].nunique():,} | Sessions: {len(sessions)}")
    print(f"Duplicate participant x session rows: {n_dup}")
    if n_dup:
        df[df.duplicated([ID, SES], keep=False)].sort_values([ID, SES]).to_csv(
            outdir / "duplicates.csv", index=False)
        print("  -> Saved to duplicates.csv. Resolve these before going further.")

    # ---------------------------------------------------------------- Step 2
    header("STEP 2. Raw values by session (including missing)")
    raw = pd.crosstab(df[SES], df[var].fillna("<NA>"), dropna=False)
    print(raw.to_string())
    raw.to_csv(outdir / "step2_raw_values_by_session.csv")
    unexpected = sorted(set(df[var].dropna()) - set(VALID_MAP))
    if unexpected:
        print(f"\nNon-0/1 values found (treated as missing): {unexpected}")
        print("  -> Check the ABCD docs for what these codes mean.")

    # ---------------------------------------------------------------- Step 3
    df["dx"] = df[var].map(VALID_MAP).astype("float")   # 0/1/NaN
    df["has_row"] = 1

    # ---------------------------------------------------------------- Step 4
    header("STEP 3. Availability and prevalence by session")
    by_ses = df.groupby(SES, observed=False).agg(
        rows=("has_row", "size"),
        valid_n=("dx", "count"),
        cases=("dx", "sum"),
    )
    by_ses["pct_valid"] = (100 * by_ses["valid_n"] / by_ses["rows"]).round(1)
    by_ses["prevalence_pct"] = (100 * by_ses["cases"] / by_ses["valid_n"]).round(2)
    by_ses["cases"] = by_ses["cases"].astype(int)
    print(by_ses.to_string())
    by_ses.to_csv(outdir / "step3_by_session.csv")
    administered = by_ses.index[by_ses["valid_n"] > 0].tolist()
    print(f"\nSessions where the variable has valid data: {administered}")
    not_admin = [s for s in sessions if s not in administered]
    if not_admin:
        print(f"Sessions with NO valid data (likely not administered): {not_admin}")

    # ---------------------------------------------------------------- Step 5
    header("STEP 4. Timepoints per participant")
    wide = df.pivot_table(index=ID, columns=SES, values="dx", aggfunc="first",
                          observed=False).reindex(columns=sessions)
    # Include participants who have rows but no valid value anywhere.
    wide = wide.reindex(df[ID].unique())
    n_tp = wide.notna().sum(axis=1)
    tp_dist = n_tp.value_counts().sort_index().rename("participants").to_frame()
    tp_dist["pct"] = (100 * tp_dist["participants"] / len(wide)).round(1)
    tp_dist.index.name = "n_valid_timepoints"
    print(tp_dist.to_string())
    tp_dist.to_csv(outdir / "step4_timepoints_per_participant.csv")
    print(f"\nParticipants with >= 2 valid timepoints (usable for change): "
          f"{(n_tp >= 2).sum():,}  |  >= 3: {(n_tp >= 3).sum():,}")

    # ---------------------------------------------------------------- Step 6
    header("STEP 5. Most common missing-data patterns (X = valid, . = missing)")
    print("Column order:", " ".join(sessions))
    patterns = wide.notna().apply(lambda r: "".join("X" if v else "." for v in r), axis=1)
    pat = patterns.value_counts().head(15).rename("participants").to_frame()
    pat["pct"] = (100 * pat["participants"] / len(wide)).round(1)
    print(pat.to_string())
    patterns.value_counts().rename("participants").to_csv(outdir / "step5_missing_patterns.csv")

    # ---------------------------------------------------------------- Step 7
    header("STEP 6. Transitions between consecutive ASSESSED visits")
    long = df.dropna(subset=["dx"]).sort_values([ID, SES])
    long["prev_dx"] = long.groupby(ID)["dx"].shift()
    long["prev_ses"] = long.groupby(ID)[SES].shift()
    tr = long.dropna(subset=["prev_dx"]).copy()
    labels = {(0, 0): "stays absent", (0, 1): "onset", (1, 0): "remits", (1, 1): "stays present"}
    tr["transition"] = [labels[(int(a), int(b))] for a, b in zip(tr["prev_dx"], tr["dx"])]
    tr["interval"] = tr["prev_ses"].astype(str) + " -> " + tr[SES].astype(str)
    trans = pd.crosstab(tr["interval"], tr["transition"], margins=True, margins_name="Total")
    trans = trans.reindex(columns=[c for c in
                                   ["stays absent", "onset", "remits", "stays present", "Total"]
                                   if c in trans.columns])
    print(trans.to_string())
    trans.to_csv(outdir / "step6_transitions.csv")
    ever = wide.max(axis=1)
    print(f"\nParticipants ever 'present' at any visit: {int((ever == 1).sum()):,} "
          f"of {int(ever.notna().sum()):,} with any valid data")
    print(f"Participants whose status ever changes: "
          f"{tr.loc[tr['transition'].isin(['onset', 'remits']), ID].nunique():,}")

    
    header("STEP 7. Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(sessions))
        ax1.bar(x, by_ses["valid_n"], color="#9db4c0", label="Valid N")
        ax1.set_ylabel("Participants with valid data")
        ax1.set_xticks(x, sessions, rotation=45, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(x, by_ses["prevalence_pct"], color="#c0392b", marker="o", label="Prevalence %")
        ax2.set_ylabel("Prevalence (%)")
        ax1.set_title(f"{var}: availability and prevalence by session")
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
        fig.tight_layout()
        fig.savefig(outdir / "step7_availability_prevalence.png", dpi=150)
        print(f"Saved {outdir / 'step7_availability_prevalence.png'}")
    except ImportError:
        print("matplotlib not installed; skipping plot.")

    print(f"\nAll tables saved in: {outdir.resolve()}")
    return {"by_session": by_ses, "timepoints": tp_dist, "patterns": pat,
            "transitions": trans, "wide": wide}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--var", default=DEFAULT_VAR)
    p.add_argument("--outdir", default="diagnostics_output")
    a = p.parse_args()
    run_diagnostics(load(a.path), a.var, a.outdir)