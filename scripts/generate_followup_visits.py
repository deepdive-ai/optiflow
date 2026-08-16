"""
One-off generator: appends a second synthetic visit for a subset of
at-risk/progressive patients in patients_examinations.csv, so the app has
longitudinal data to compare (Iteration 4) and later detect change in
(Iteration 6).

Selection: patients who are Referral_Required == "Yes", or whose
Assessment_Category is inherently progressive (Myopia Progression), on the
theory that flagged/progressive patients are the ones a real clinic calls
back for monitoring. Values evolve with a category-driven directional trend
plus random noise, not pure noise, so a before/after comparison shows a
plausible clinical signal.

Run once: python scripts/generate_followup_visits.py
Re-running is a no-op if follow-up visits already exist (detected via
duplicate Patient_ID count > 1).
"""

import random
from datetime import timedelta
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "patients_examinations.csv"
SEED = 42

VA_SCALE = ["6/6", "6/9", "6/12", "6/18", "6/24"]

CATEGORY_TARGETS = {
    ("Suspected Glaucoma", "Yes"): 24,
    ("Cataract", "Yes"): 6,
    ("Keratoconus", "Yes"): 3,
    ("Myopia Progression", "No"): 4,
}


def step_va(value: str, steps: int) -> str:
    """Move `value` up/down the Snellen scale by `steps` lines (positive = worse), clipped to range."""
    idx = VA_SCALE.index(value)
    idx = max(0, min(len(VA_SCALE) - 1, idx + steps))
    return VA_SCALE[idx]


def evolve_glaucoma(row: pd.Series, rng: random.Random) -> dict:
    iop_od = row["IOP_OD_mmHg"] + rng.choice([1, 2, 2, 3, -1])
    iop_os = row["IOP_OS_mmHg"] + rng.choice([1, 2, 2, 3, -1])
    cd_od = round(min(0.9, row["CD_Ratio_OD"] + rng.uniform(0.01, 0.05)), 2)
    cd_os = round(min(0.9, row["CD_Ratio_OS"] + rng.uniform(0.01, 0.05)), 2)
    va_od = step_va(row["Visual_Acuity_OD"], rng.choice([0, 0, 1]))
    va_os = step_va(row["Visual_Acuity_OS"], rng.choice([0, 0, 1]))
    return {
        "IOP_OD_mmHg": max(10, min(35, iop_od)),
        "IOP_OS_mmHg": max(10, min(35, iop_os)),
        "CD_Ratio_OD": cd_od,
        "CD_Ratio_OS": cd_os,
        "Visual_Acuity_OD": va_od,
        "Visual_Acuity_OS": va_os,
        "Spherical_Equivalent_OD_D": round(row["Spherical_Equivalent_OD_D"] + rng.uniform(-0.2, 0.2), 2),
        "Spherical_Equivalent_OS_D": round(row["Spherical_Equivalent_OS_D"] + rng.uniform(-0.2, 0.2), 2),
    }


def evolve_cataract(row: pd.Series, rng: random.Random) -> dict:
    va_od = step_va(row["Visual_Acuity_OD"], rng.choice([1, 1, 2]))
    va_os = step_va(row["Visual_Acuity_OS"], rng.choice([1, 1, 2]))
    return {
        "IOP_OD_mmHg": max(10, min(35, row["IOP_OD_mmHg"] + rng.choice([-1, 0, 0, 1]))),
        "IOP_OS_mmHg": max(10, min(35, row["IOP_OS_mmHg"] + rng.choice([-1, 0, 0, 1]))),
        "CD_Ratio_OD": row["CD_Ratio_OD"],
        "CD_Ratio_OS": row["CD_Ratio_OS"],
        "Visual_Acuity_OD": va_od,
        "Visual_Acuity_OS": va_os,
        "Spherical_Equivalent_OD_D": round(row["Spherical_Equivalent_OD_D"] + rng.uniform(0, 0.4), 2),
        "Spherical_Equivalent_OS_D": round(row["Spherical_Equivalent_OS_D"] + rng.uniform(0, 0.4), 2),
    }


def evolve_keratoconus(row: pd.Series, rng: random.Random) -> dict:
    va_od = step_va(row["Visual_Acuity_OD"], rng.choice([0, 1, 1]))
    va_os = step_va(row["Visual_Acuity_OS"], rng.choice([0, 1, 1]))
    return {
        "IOP_OD_mmHg": row["IOP_OD_mmHg"],
        "IOP_OS_mmHg": row["IOP_OS_mmHg"],
        "CD_Ratio_OD": row["CD_Ratio_OD"],
        "CD_Ratio_OS": row["CD_Ratio_OS"],
        "Visual_Acuity_OD": va_od,
        "Visual_Acuity_OS": va_os,
        "Spherical_Equivalent_OD_D": round(row["Spherical_Equivalent_OD_D"] - rng.uniform(0.5, 1.5), 2),
        "Spherical_Equivalent_OS_D": round(row["Spherical_Equivalent_OS_D"] - rng.uniform(0.5, 1.5), 2),
    }


def evolve_myopia_progression(row: pd.Series, rng: random.Random) -> dict:
    va_od = step_va(row["Visual_Acuity_OD"], rng.choice([0, 0, 1]))
    va_os = step_va(row["Visual_Acuity_OS"], rng.choice([0, 0, 1]))
    return {
        "IOP_OD_mmHg": row["IOP_OD_mmHg"],
        "IOP_OS_mmHg": row["IOP_OS_mmHg"],
        "CD_Ratio_OD": row["CD_Ratio_OD"],
        "CD_Ratio_OS": row["CD_Ratio_OS"],
        "Visual_Acuity_OD": va_od,
        "Visual_Acuity_OS": va_os,
        "Spherical_Equivalent_OD_D": round(row["Spherical_Equivalent_OD_D"] - rng.uniform(0.25, 1.0), 2),
        "Spherical_Equivalent_OS_D": round(row["Spherical_Equivalent_OS_D"] - rng.uniform(0.25, 1.0), 2),
    }


EVOLVERS = {
    "Suspected Glaucoma": evolve_glaucoma,
    "Cataract": evolve_cataract,
    "Keratoconus": evolve_keratoconus,
    "Myopia Progression": evolve_myopia_progression,
}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["Visit_Date"] = pd.to_datetime(df["Visit_Date"])

    if df["Patient_ID"].duplicated().any():
        print("Follow-up visits already present (duplicate Patient_IDs found) — skipping.")
        return

    rng = random.Random(SEED)

    selected_ids = []
    for (category, referral_status), target_n in CATEGORY_TARGETS.items():
        pool = df[
            (df["Assessment_Category"] == category) & (df["Referral_Required"] == referral_status)
        ]
        n = min(target_n, len(pool))
        chosen = pool.sample(n=n, random_state=SEED)
        selected_ids.extend(chosen["Patient_ID"].tolist())

    followup_rows = []
    for patient_id in selected_ids:
        original = df[df["Patient_ID"] == patient_id].iloc[0]
        evolve = EVOLVERS[original["Assessment_Category"]]
        changes = evolve(original, rng)

        new_row = original.copy()
        new_row["Visit_Date"] = original["Visit_Date"] + timedelta(days=rng.randint(90, 270))
        for field, value in changes.items():
            new_row[field] = value
        followup_rows.append(new_row)

    followups_df = pd.DataFrame(followup_rows)
    combined = pd.concat([df, followups_df], ignore_index=True)
    combined = combined.sort_values(["Patient_ID", "Visit_Date"]).reset_index(drop=True)
    combined["Visit_Date"] = combined["Visit_Date"].dt.strftime("%Y-%m-%d")

    combined.to_csv(CSV_PATH, index=False)
    print(f"Added {len(followup_rows)} follow-up visits for {len(selected_ids)} patients.")
    print(f"Total rows: {len(combined)} (was {len(df)}).")


if __name__ == "__main__":
    main()
