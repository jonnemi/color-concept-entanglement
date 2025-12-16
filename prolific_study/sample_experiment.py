import argparse
import random
import pandas as pd
from pathlib import Path

PCTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def sample_experiment_1(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)

    used_objects = set()
    rows = []

    for pct in PCTS:
        candidates = df[
            (df["percent_colored"] == pct) &
            (~df["object"].isin(used_objects))
        ]

        if len(candidates) == 0:
            raise RuntimeError(
                f"No available images for {pct}% with unused objects."
            )

        chosen = candidates.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0]
        rows.append(chosen)
        used_objects.add(chosen["object"])

    sampled = pd.DataFrame(rows)

    # shuffle trial order
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimulus-table", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.stimulus_table)

    sampled = sample_experiment_1(df, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.out, index=False)

    print(f"Sampled {len(sampled)} trials")
    print(sampled[["object", "percent_colored"]])


if __name__ == "__main__":
    main()
