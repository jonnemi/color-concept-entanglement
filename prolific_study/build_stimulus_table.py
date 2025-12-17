import re
from pathlib import Path
import pandas as pd


# Variant label helper
def variant_label(p: Path) -> str:
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
# Example:
# acoustic_guitar_1_ab533cc0_resized_purple
FOLDER_RE = re.compile(
    r"^(?P<object>.+?)_(?P<instance>\d+)_(?P<hash>[a-f0-9]+)_resized_(?P<color>[a-z]+)$"
)

def parse_folder_name(name: str) -> dict:
    """
    Parse dataset folder name and extract object + target_color.
    """
    m = FOLDER_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized folder name: {name}")

    return {
        "object": m.group("object"),
        "target_color": m.group("color"),
    }


# Table builder
VARIANT_RE = re.compile(r"(FG|BG)_(\d{3})_(ind|seq)\.png")

def build_stimulus_table(
    dataset_root: Path,
    stimulus_type: str
) -> pd.DataFrame:
    """
    Build a stimulus table from a dataset directory.

    Args:
        dataset_root: Path to directory containing object/shape folders
        stimulus_type: "correct_prior", "counterfact", or "shape"

    Returns:
        pd.DataFrame with one row per stimulus image
    """

    if stimulus_type not in {"correct_prior", "counterfact", "shape"}:
        raise ValueError("stimulus_type must be 'correct_prior', 'counterfact', or 'shape'")
    rows = []

    dataset_root = Path(dataset_root)

    for obj_dir in sorted(dataset_root.iterdir()):
        if not obj_dir.is_dir():
            continue

        meta = parse_folder_name(obj_dir.name)

        for img in sorted(obj_dir.glob("*.png")):
            m = VARIANT_RE.match(img.name)

            if m:
                region, pct, mode = m.groups()
                percent_colored = int(pct)
                variant_region = region
            else:
                # base / white image
                percent_colored = 0
                variant_region = "white"
                mode = "base"

            rows.append({
                "object": meta["object"].replace("_", " "),
                "stimulus_type": stimulus_type,

                # color used for manipulation
                "target_color": meta["target_color"],

                # manipulation metadata
                "variant_region": variant_region,
                "percent_colored": percent_colored,
                "mode": mode,
                "variant_label": variant_label(img),

                # path for jsPsych
                "image_path": str(img.relative_to(dataset_root)),
            })


    return pd.DataFrame(rows)
