import argparse
import random
import json
from pathlib import Path
import pandas as pd


# CONFIG
SANITY_POSITIONS = [5, 25, 45, 65, 85]  # 1-based indexing
TOTAL_QUESTIONS = 106

# Percent bins
PCTS_13 = [0, 5, 10, 20, 30, 40, 50, 55, 60, 70, 80, 90, 100]
PCTS_12_NO_ZERO = [5, 10, 20, 30, 40, 50, 55, 60, 70, 80, 90, 100]
PCTS_5_BG = [20, 40, 60, 80, 100]


# HELPERS
def _sample_unique(df, rng, n):
    if len(df) < n:
        raise RuntimeError("Not enough candidates to sample from.")
    return df.sample(n=n, random_state=rng.randint(0, 10**9)).to_dict("records")


# OBJECT SAMPLERS
def sample_counterfactual_objects(df, rng):
    rows = []
    used_objects = set()

    for pct in PCTS_12_NO_ZERO:
        candidates = df[
            (df["percent_colored"] == pct) &
            (df["condition"] == "counterfact") &
            (~df["object"].isin(used_objects))
        ]
        chosen = _sample_unique(candidates, rng, 1)[0]
        used_objects.add(chosen["object"])
        rows.append(chosen)

    return rows


def sample_background_objects(df, rng):
    rows = []
    used_objects = set()

    for pct in PCTS_5_BG:
        candidates = df[
            (df["percent_colored"] == pct) &
            (df["condition"] == "background") &
            (~df["object"].isin(used_objects))
        ]
        chosen = _sample_unique(candidates, rng, 1)[0]
        used_objects.add(chosen["object"])
        rows.append(chosen)

    return rows


def sample_priors_objects(df, rng):
    rows = []

    for pct in PCTS_13:
        candidates = df[
            (df["percent_colored"] == pct) &
            (df["condition"] == "priors")
        ]
        chosen = _sample_unique(candidates, rng, 3)
        rows.extend(chosen)

    return rows


# SHAPE SAMPLERS
def sample_background_shapes(df, rng):
    rows = []
    used_shapes = set()

    for pct in PCTS_5_BG:
        candidates = df[
            (df["percent_colored"] == pct) &
            (df["condition"] == "background") &
            (~df["object"].isin(used_shapes))
        ]
        chosen = _sample_unique(candidates, rng, 1)[0]
        used_shapes.add(chosen["object"])
        rows.append(chosen)

    return rows


def sample_priors_shapes(df, rng):
    rows = []

    for pct in PCTS_13:
        candidates = df[
            (df["percent_colored"] == pct) &
            (df["condition"] == "priors")
        ]
        chosen = _sample_unique(candidates, rng, 3)
        rows.extend(chosen)

    return rows


# FIXED QUESTIONS
def make_sanity_question(idx):
    return {
        "question_type": "sanity",
        "sanity_id": idx
    }


def make_introspection_question():
    return {
        "question_type": "introspection",
        "prompt": (
            "For any object, x% of its pixels should be colored "
            "for it to be considered that color. What value should x% be?"
        ),
        "min": 0,
        "max": 100
    }


# PROFILE GENERATION
def generate_profile(df_objects, df_shapes, seed, introspection_position):
    rng = random.Random(seed)

    questions = []

    # --- Objects ---
    questions += sample_counterfactual_objects(df_objects, rng)
    questions += sample_background_objects(df_objects, rng)
    questions += sample_priors_objects(df_objects, rng)

    # --- Shapes ---
    questions += sample_background_shapes(df_shapes, rng)
    questions += sample_priors_shapes(df_shapes, rng)

    if len(questions) != 100:
        raise RuntimeError(f"Expected 100 variable questions, got {len(questions)}")

    # Shuffle all variable questions together
    rng.shuffle(questions)

    # Insert sanity checks (1-based positions)
    for pos in sorted(SANITY_POSITIONS):
        questions.insert(pos - 1, make_sanity_question(pos))

    # Insert introspection
    introspection = make_introspection_question()
    if introspection_position == "first":
        questions.insert(0, introspection)
    elif introspection_position == "last":
        questions.append(introspection)
    else:
        raise ValueError("introspection_position must be 'first' or 'last'")

    if len(questions) != TOTAL_QUESTIONS:
        raise RuntimeError(
            f"Final survey length mismatch: {len(questions)} != {TOTAL_QUESTIONS}"
        )

    return questions


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-table", type=Path, required=True)
    parser.add_argument("--shapes-table", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--introspection", choices=["first", "last"], required=True)
    parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    df_objects = pd.read_csv(args.objects_table)
    df_shapes = pd.read_csv(args.shapes_table)

    profile = generate_profile(
        df_objects=df_objects,
        df_shapes=df_shapes,
        seed=args.seed,
        introspection_position=args.introspection
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Wrote survey profile with {len(profile)} questions → {args.out}")


if __name__ == "__main__":
    main()
