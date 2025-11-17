from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats

def normalize_colors(c):
    """Normalize color entries to lowercase string list."""
    if c is None:
        return []
    if isinstance(c, str):
        return [c.strip().lower()]
    try:
        return [str(x).strip().lower() for x in c if str(x).strip()]
    except Exception:
        return [str(c).strip().lower()]


def _variant_sort_key(p: Path):
    """Sort variants in FG/BG order"""
    name = p.name
    m_fg = re.match(r"FG_(\d{3})\.png$", name)
    if m_fg:
        return (0, int(m_fg.group(1)))
    m_bg = re.match(r"BG_(\d{3})\.png$", name)
    if m_bg:
        return (1, int(m_bg.group(1)))
    return (9, name)


def collect_variants_for(image_path: str, target_color: str, out_root: Path) -> list[Path]:
    """
    Collect all FG_*.png and BG_*.png variant images for a given base image and color.
    """
    stem = Path(image_path).stem
    color_dir = out_root / f"{stem}_{target_color}"
    if not color_dir.exists():
        return []

    fg = list(color_dir.glob("FG_*.png"))
    bg = list(color_dir.glob("BG_*.png"))
    paths = fg + bg
    return sorted(paths, key=_variant_sort_key)


def show_variants_grid(
    image_path: str,
    target_color: str,
    out_root: Path,
    df_predictions: pd.DataFrame | None = None,
    question: str = "",    # empty, "this" or "most"
    thumb_w: int = 256,
    row_h: float = 3.0,
    fontsize: int = 14,
    color_mode : str = "independent"
):
    """
    Display a 2-row grid of FG/BG color variants for a given image.

    If df_predictions is provided, also show predicted colors (pred_color_this / pred_color_most)
    beneath each variant image.

    Parameters
    ----------
    image_path : str or Path
        Base image path (used to find variant subfolder).
    target_color : str
        Target recolor name.
    out_root : Path
        Root directory containing the generated variants.
    df_predictions : pd.DataFrame, optional
        DataFrame with predicted colors. Must include columns:
          - image_variant
          - correct_answer
          - pred_color_this / pred_color_most
    question : str
        "this" or "most" - determines which prediction column to use.
    thumb_w : int
        Thumbnail width per image.
    row_h : float
        Row height scaling factor.
    fontsize : int
        Font size for titles and predictions.
    color_mode : str
        "independent" or "sequential" - indicates which variant type to display.
    """

    # Load variant paths
    paths = collect_variants_for(image_path, target_color, out_root)
    if not paths:
        print(f"No variants found for {image_path}")
        return

    mode_suffix = "ind" if color_mode == "independent" else "seq"

    fg_paths = [p for p in paths if p.name.startswith("FG_") and p.name.endswith(f"{mode_suffix}.png")]
    bg_paths = [p for p in paths if p.name.startswith("BG_") and p.name.endswith(f"{mode_suffix}.png")]

    cols = len(fg_paths)
    rows = 2

    # Figure sizing
    fig_w = cols * (thumb_w / 80)
    fig_h = rows * row_h
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # Ensure consistent axis shape
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[axes[0]], [axes[1]]]

    # Helper to get model prediction (if df_predictions is provided)
    def get_prediction(label):
        if df_predictions is None:
            return None

        pred_col = "pred_color_this" if question.lower() == "this" else "pred_color_most"
        match = df_predictions[
            (df_predictions["image_variant"].str.lower() == label.lower())
            & (df_predictions["correct_answer"].str.lower() == target_color.lower())
        ]
        if not match.empty and pred_col in match.columns:
            return str(match.iloc[0][pred_col])
        return None

    def _draw_row(ax_row, paths_row, title_prefix, start_col=0):
        for c in range(cols):
            ax = ax_row[c]
            ax.axis("off")
            idx = c - start_col
            if 0 <= idx < len(paths_row):
                p = paths_row[idx]
                im = Image.open(p).convert("RGB")
                ax.imshow(im)

                m = re.search(r"(\d{3})(?=\.png$)", p.name)
                pct = int(m.group(1)) if m else ""
                label = f"{title_prefix} {pct}%" if pct != "" else title_prefix
                pred = get_prediction(label)

                ax.set_title(label, fontsize=fontsize, pad=4)
                if pred:
                    ax.text(
                        0.5, -0.15, pred,
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=fontsize,
                        fontweight="bold"
                    )

    _draw_row(axes[0], fg_paths, "FG", start_col=0)
    _draw_row(axes[1], bg_paths, "BG", start_col=1)
    
    if not question == "":
        question = f", question: {question}"

    fig.suptitle(
        f"{Path(image_path).name} - target: {target_color}, color_mode: {color_mode}{question}",
        fontsize=fontsize + 4,
        fontweight="bold"
    )

    plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4, wspace=0.05)
    plt.show()


def plot_vlm_performance(
    df,
    show_accuracy=True,
    show_probability=True,
    ci=True, 
):
    """
    Plot model performance (accuracy and/or P(correct)) vs. recoloring fraction with error bars.

    Expects columns:
      ['image_variant', 'correct_answer', 'pred_color_this', 'pred_color_most',
       'prob_correct_this', 'prob_correct_most']

    Args:
        df (pd.DataFrame): model results
        show_accuracy (bool): include accuracy curves
        show_probability (bool): include probability curves
        ci (bool): use 95% confidence interval instead of std deviation
    """

    def parse_variant(variant):
        """Extract FG/BG and numeric percentage."""
        m = re.match(r"(FG|BG)\s*(\d+)%", str(variant))
        if not m:
            return None, None
        kind, pct = m.groups()
        return kind, int(pct)

    # --- Parse FG/BG and % ---
    df[["region", "pct"]] = df["image_variant"].apply(lambda v: pd.Series(parse_variant(v)))
    df = df.dropna(subset=["region", "pct"])

    # --- Compute binary accuracy ---
    if "pred_color_this" in df.columns:
        df["acc_this"] = (df["pred_color_this"].str.lower() == df["correct_answer"].str.lower()).astype(float)
    if "pred_color_most" in df.columns:
        df["acc_most"] = (df["pred_color_most"].str.lower() == df["correct_answer"].str.lower()).astype(float)

    # --- Group by recoloring level and region ---
    grouped = df.groupby(["region", "pct"])

    # Helper to compute mean and error
    def summarize(col):
        mean = grouped[col].mean()
        std = grouped[col].std()
        n = grouped[col].count()
        if ci:
            # 95% confidence interval
            ci_val = stats.t.ppf(0.975, n - 1) * (std / np.sqrt(n))
            return mean, ci_val
        else:
            return mean, std

    # Collect stats
    data = {}
    metrics = []
    if show_accuracy:
        metrics += ["acc_this", "acc_most"]
    if show_probability:
        metrics += ["prob_correct_this", "prob_correct_most"]

    for metric in metrics:
        if metric not in df.columns:
            continue
        mean, err = summarize(metric)
        data[f"{metric}_mean"] = mean
        data[f"{metric}_err"] = err

    summary = pd.DataFrame(data).reset_index()

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {
        "this_FG": "#1f77b4",
        "this_BG": "#ff7f0e",
        "most_FG": "#2ca02c",
        "most_BG": "#d62728",
    }

    def plot_metric(region, acc_col, prob_col, label_prefix):
        sub = summary[summary["region"] == region]
        if sub.empty:
            return
        # Accuracy = dashed
        if show_accuracy:
            ax.errorbar(
                sub["pct"], sub[f"{acc_col}_mean"], yerr=sub[f"{acc_col}_err"],
                fmt="o--", capsize=3, color=colors[f"{label_prefix}_{region}"],
                label=f"{label_prefix}_{region} (accuracy)", alpha=0.8
            )
        # Probability = solid
        if show_probability:
            ax.errorbar(
                sub["pct"], sub[f"{prob_col}_mean"], yerr=sub[f"{prob_col}_err"],
                fmt="o-", capsize=3, color=colors[f"{label_prefix}_{region}"],
                label=f"{label_prefix}_{region} (P(correct))", alpha=0.9
            )

    for region in ["FG", "BG"]:
        plot_metric(region, "acc_this", "prob_correct_this", "this")
        plot_metric(region, "acc_most", "prob_correct_most", "most")

    ax.set_xlabel("Colored pixel percentage (%)", fontsize=12)
    ax.set_ylabel("Accuracy / P(correct)", fontsize=12)
    ax.set_title("VLM performance vs. recoloring fraction", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(0, 110, 10))
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Question / Region", fontsize=10)

    plt.tight_layout()
    plt.show()
