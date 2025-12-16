import argparse
import re
from pathlib import Path
import pandas as pd


# Variant label helper
def variant_label(p: Path):
    """
    Create readable labels from variant filenames:
    FG_030_seq.png  -> "FG 30% (seq)"
    BG_050_ind.png  -> "BG 50% (ind)"
    base image      -> "white"
    """
    stem = p.stem
    m = re.match(r"(FG|BG)_(\d{3})_(ind|seq)$", stem)
    if m:
        region, pct, mode = m.groups()
        return f"{region} {int(pct)}% ({mode})"
    return "white"


# Folder-name parser
FOLDER_RE = re.compile(
    r"^(?P<object>.+?)_(?P<instance>\d+)_(?P<hash>[a-f0-9]+)_resized_(?P<color>[a-z]+)$"
)

def parse_folder_name(name: str):
    m = FOLDER_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized folder name: {name}")

    return {
        "object": m.group("object"),
        "correct_color": m.group("color"),
    }


# Table builder
VARIANT_RE = re.compile(r"(FG|BG)_(\d{3})_(ind|seq)\.png")

def build_stimulus_table(dataset_root: Path) -> pd.DataFrame:
    rows = []

    for obj_dir in sorted(dataset_root.iterdir()):
        if not obj_dir.is_dir():
            continue

        meta = parse_folder_name(obj_dir.name)

        for img in sorted(obj_dir.glob("*.png")):
            m = VARIANT_RE.match(img.name)

            if m:
                region, pct, mode = m.groups()
                rows.append({
                    **meta,
                    "variant_region": region,
                    "percent_colored": int(pct),
                    "mode": mode,
                    "variant_label": variant_label(img),
                    "image_path": str(img.relative_to(dataset_root)),
                })
            else:
                # base / white image
                rows.append({
                    **meta,
                    "variant_region": "white",
                    "percent_colored": 0,
                    "mode": "base",
                    "variant_label": "white",
                    "image_path": str(img.relative_to(dataset_root)),
                })

    return pd.DataFrame(rows)


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = build_stimulus_table(args.dataset_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    df.to_parquet(args.out.with_suffix(".parquet"))

    print(f"Wrote {len(df)} stimuli")
    print(args.out.with_suffix(".csv"))
    print(args.out.with_suffix(".parquet"))


if __name__ == "__main__":
    main()
