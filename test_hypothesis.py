#!/usr/bin/env python3
"""
Test Hypothesis: emb(H2, V3, C2) + (emb(H1, V1, C1) - emb(H1, V2, C1)) ≈ emb(H2, V3 + (V1 - V2), C2)

This tests if value differences from one hue can be linearly transferred to another hue.
Visualization with color swatches in PDF.
"""

import sys

sys.path.insert(0, "/Users/georgij/Documents/Работа/Color-MMLs")

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import matplotlib

matplotlib.use("Agg")

from utils.analyze.munsell_analyze import MunsellEmbeddingsAnalyzer


def get_embedding_by_hvc(meta_df, embeddings, h, v, c):
    """Find embedding index for given H, V, C combination."""
    mask = (meta_df["H"] == h) & (meta_df["V"] == v) & (meta_df["C"] == c)
    indices = meta_df[mask].index.tolist()
    if len(indices) == 0:
        return None, None
    return indices[0], embeddings[indices[0]]


def test_hypothesis(embeddings, meta_df, n_tests=500, seed=42):
    """Test the linearity hypothesis across multiple color combinations."""
    np.random.seed(seed)

    results = []

    # Group colors by H and C to find valid V values
    grouped = meta_df.groupby(["H", "C"])["V"].apply(list).to_dict()

    valid_hc = [(h, c) for (h, c), vs in grouped.items() if len(vs) >= 2]
    all_h = meta_df["H"].unique()
    all_c = meta_df["C"].unique()
    all_v = sorted(meta_df["V"].unique())

    for _ in range(n_tests):
        # Select H1, C1 (source)
        h1, c1 = valid_hc[np.random.randint(len(valid_hc))]
        v_values = grouped[(h1, c1)]
        v1, v2 = sorted(np.random.choice(v_values, 2, replace=False))

        # Select H2, C2 (target)
        h2, c2 = valid_hc[np.random.randint(len(valid_hc))]

        # Select V3 (base value for target)
        v3 = np.random.choice(grouped[(h2, c2)])

        # Calculate target V: V3 + (V1 - V2)
        target_v = v3 + (v1 - v2)

        # Get embeddings
        idx1, emb1 = get_embedding_by_hvc(meta_df, embeddings, h1, v1, c1)
        idx2, emb2 = get_embedding_by_hvc(meta_df, embeddings, h1, v2, c1)
        idx3, emb3 = get_embedding_by_hvc(meta_df, embeddings, h2, v3, c2)

        if any(x is None for x in [emb1, emb2, emb3]):
            continue

        # Compute predicted: emb(H2, V3, C2) + (emb(H1, V1, C1) - emb(H1, V2, C1))
        delta = emb1 - emb2
        predicted = emb3 + delta

        # Find target embedding if it exists
        idx_target, emb_target = get_embedding_by_hvc(
            meta_df, embeddings, h2, target_v, c2
        )

        # Find nearest embedding to predicted
        similarities = []
        for i, emb in enumerate(embeddings):
            if i in [idx1, idx2, idx3]:
                continue
            sim = 1 - cosine(predicted, emb)
            similarities.append((i, sim, meta_df.iloc[i]))

        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_idx, nearest_sim, nearest_meta = similarities[0]

        # Record result
        result = {
            "h1": h1,
            "v1": v1,
            "c1": c1,
            "h2": h2,
            "v2": v2,
            "v3": v3,
            "c2": c2,
            "target_v": target_v,
            "target_exists": idx_target is not None,
            "nearest_h": nearest_meta["H"],
            "nearest_v": nearest_meta["V"],
            "nearest_c": nearest_meta["C"],
            "nearest_sim": nearest_sim,
            "delta_v": v1 - v2,
            # Store RGB for visualization
            "rgb_source": meta_df.iloc[idx1]["RGB"] if idx1 is not None else None,
            "rgb_base": meta_df.iloc[idx3]["RGB"] if idx3 is not None else None,
            "rgb_nearest": nearest_meta["RGB"],
        }

        if idx_target is not None:
            result["target_sim"] = 1 - cosine(predicted, emb_target)
            result["target_dist"] = euclidean(predicted, emb_target)
            result["nearest_is_target"] = nearest_idx == idx_target
            result["rgb_target"] = meta_df.iloc[idx_target]["RGB"]
            result["target_idx"] = idx_target
            result["nearest_idx"] = nearest_idx
            result["idx1"] = idx1
            result["idx3"] = idx3

        results.append(result)

    return pd.DataFrame(results)


def analyze_results(results_df, name):
    """Analyze and print results."""
    print(f"\n{'=' * 60}")
    print(f"Results for {name}")
    print(f"{'=' * 60}")

    print(f"\nTotal tests: {len(results_df)}")
    print(f"Target existed: {results_df['target_exists'].sum()}")

    # Filter to cases where target exists
    valid = results_df[results_df["target_exists"]]
    if len(valid) == 0:
        print("No valid target cases found")
        return

    print(f"\nTarget exists cases: {len(valid)}")

    # Accuracy: nearest is target
    if "nearest_is_target" in valid.columns:
        accuracy = valid["nearest_is_target"].mean()
        print(f"Nearest is target: {accuracy:.2%}")

    # Cosine similarity to target
    if "target_sim" in valid.columns:
        mean_sim = valid["target_sim"].mean()
        std_sim = valid["target_sim"].std()
        print(f"Target similarity: {mean_sim:.4f} ± {std_sim:.4f}")

        # Distribution
        print(f"  Min: {valid['target_sim'].min():.4f}")
        print(f"  25%: {valid['target_sim'].quantile(0.25):.4f}")
        print(f"  Median: {valid['target_sim'].median():.4f}")
        print(f"  75%: {valid['target_sim'].quantile(0.75):.4f}")
        print(f"  Max: {valid['target_sim'].max():.4f}")

    # Euclidean distance
    if "target_dist" in valid.columns:
        mean_dist = valid["target_dist"].mean()
        std_dist = valid["target_dist"].std()
        print(f"Target distance: {mean_dist:.4f} ± {std_dist:.4f}")

    # Correlation with delta_V
    if "target_sim" in valid.columns:
        corr, p = pearsonr(valid["delta_v"], valid["target_sim"])
        print(f"\nCorrelation (|ΔV| vs target_sim): r={corr:.4f}, p={p:.2e}")

    return valid


def visualize_examples(
    results_df, meta_df, n_examples=12, filename="hypothesis_visualization.pdf"
):
    """Create PDF with color swatches showing prediction vs ground truth."""
    import matplotlib.backends.backend_pdf as pdf_backend

    # Filter valid results
    valid = results_df[results_df["target_exists"]].copy()
    if len(valid) == 0:
        print("No valid results to visualize")
        return

    # Sort by similarity to show worst cases first
    valid = valid.sort_values("target_sim", ascending=True).head(n_examples)

    # Create PDF
    pdf_pages = pdf_backend.PdfPages(filename)

    # Create figure for scatter plot: delta_v vs similarity
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(valid["delta_v"], valid["target_sim"], alpha=0.6, s=50, c="steelblue")

    # Add regression line
    z = np.polyfit(valid["delta_v"], valid["target_sim"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid["delta_v"].min(), valid["delta_v"].max(), 100)
    ax1.plot(
        x_line,
        p(x_line),
        "r--",
        linewidth=2,
        label=f"r={pearsonr(valid['delta_v'], valid['target_sim'])[0]:.3f}",
    )

    ax1.set_xlabel("ΔV (Value Difference)", fontsize=12)
    ax1.set_ylabel("Target Similarity (Cosine)", fontsize=12)
    ax1.set_title("ΔV vs Prediction Similarity", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    pdf_pages.savefig(fig1)
    plt.close(fig1)

    # Create color swatches figure
    n_cols = 4
    n_rows = int(np.ceil(n_examples / n_cols))

    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    fig2.suptitle(
        "Color Prediction Examples\nC1 + (C2 - C1) = Predicted → Closest vs GT",
        fontsize=14,
        y=1.02,
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, row) in enumerate(valid.iterrows()):
        ax = axes[idx // n_cols, idx % n_cols]

        # Get colors: emb1 (H1,V1,C1), emb2 (H1,V2,C1), emb3 (H2,V3,C2), nearest, target
        rgb_emb1 = row.get("rgb_source", None)  # emb(H1, V1, C1)
        # Get RGB for emb2 (H1, V2, C1) by looking up in meta_df
        emb2_row = meta_df[
            (meta_df["H"] == row["h1"])
            & (meta_df["V"] == row["v2"])
            & (meta_df["C"] == row["c1"])
        ]
        rgb_emb2 = (
            emb2_row.iloc[0]["RGB"] if len(emb2_row) > 0 else None
        )  # emb(H1, V2, C1)
        rgb_emb3 = row.get("rgb_base", None)  # emb(H2, V3, C2)
        rgb_nearest = row.get("rgb_nearest", None)
        rgb_target = row.get("rgb_target", None)

        # Draw 5 color squares
        box_width = 0.85
        box_height = 0.14
        start_y = 0.95 - box_height

        # 5 colors: emb1, emb2, emb3 (base), nearest, target
        colors_to_show = [
            (rgb_emb1, f"C1: {row['h1']} V{row['v1']} C{row['c1']}"),
            (rgb_emb2, f"C2: {row['h1']} V{row['v2']} C{row['c1']}"),
            (rgb_emb3, f"Base: {row['h2']} V{row['v3']} C{row['c2']}"),
            (
                rgb_nearest,
                f"Nearest: {row['nearest_h']} V{row['nearest_v']} C{row['nearest_c']}",
            ),
            (
                rgb_target,
                f"GT Target: {row['h2']} V{int(row['target_v'])} C{row['c2']}",
            ),
        ]

        y_positions = [
            start_y - i * (box_height + 0.02) for i in range(len(colors_to_show))
        ]

        for i, (rgb, label) in enumerate(colors_to_show):
            if rgb is not None:
                # Normalize RGB if needed
                if isinstance(rgb, (tuple, list)):
                    if max(rgb) > 1:
                        rgb = tuple([c / 255 for c in rgb])
                rect = patches.Rectangle(
                    (0.05, y_positions[i]),
                    box_width,
                    box_height,
                    facecolor=rgb,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)
        for i, (rgb, label) in enumerate(colors_to_show):
            if rgb is not None:
                # Normalize RGB if needed
                if isinstance(rgb, (tuple, list)):
                    if max(rgb) > 1:
                        rgb = tuple([c / 255 for c in rgb])
                rect = patches.Rectangle(
                    (0.05, y_positions[i]),
                    box_width,
                    box_height,
                    facecolor=rgb,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)
                ax.text(
                    0.5,
                    y_positions[i] + box_height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=ax.transAxes,
                    fontweight="bold" if i >= 3 else "normal",
                )

        # Show similarity score
        target_sim = row.get("target_sim", 0)
        ax.text(
            0.5,
            -0.05,
            f"Similarity: {target_sim:.4f}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(valid), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.tight_layout()
    pdf_pages.savefig(fig2)
    plt.close(fig2)

    pdf_pages.close()
    print(f"Visualization saved to {filename}")


def main():
    print("Loading embeddings...")
    analyzer = MunsellEmbeddingsAnalyzer(
        "data/embeddings/qwen2.5_7B/munsell_colors_describe",
        "data/colors/munsell_colors/munsell_manifest.csv",
    )

    embeddings_dict = analyzer.chain_loader.get_all_available_embeddings()
    metadata = analyzer.chain_loader.get_all_available_colors()

    print(f"Loaded {len(metadata)} samples")

    # Prepare data
    meta_df = pd.DataFrame(metadata)
    vl_embeds = np.array(embeddings_dict["vl_pooled"])
    lm_embeds = np.array(embeddings_dict["lm_pooled"])

    # Filter greys
    vl_filtered, meta_filtered = analyzer.filter_embeddings_leave_one_grey(
        vl_embeds, metadata
    )
    lm_filtered, _ = analyzer.filter_embeddings_leave_one_grey(lm_embeds, metadata)
    meta = meta_filtered.reset_index(drop=True)

    print(f"Filtered: {len(meta)} samples")
    print(
        f"Unique H: {meta['H'].nunique()}, V: {meta['V'].nunique()}, C: {meta['C'].nunique()}"
    )

    # Test hypothesis for both embedding types
    n_tests = 500

    print("\n" + "=" * 70)
    print(
        "TESTING HYPOTHESIS: emb(H2,V3,C2) + (emb(H1,V1,C1)-emb(H1,V2,C1)) ≈ emb(H2,V3+(V1-V2),C2)"
    )
    print("=" * 70)

    # VL embeddings
    vl_results = test_hypothesis(vl_filtered, meta, n_tests=n_tests)
    vl_valid = analyze_results(vl_results, "VL (Vision-Language)")

    # LM embeddings
    lm_results = test_hypothesis(lm_filtered, meta, n_tests=n_tests)
    lm_valid = analyze_results(lm_results, "LM (Language Model)")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    if "target_sim" in vl_valid.columns and "target_sim" in lm_valid.columns:
        print(f"\n{'Metric':<30} {'VL':>15} {'LM':>15}")
        print("-" * 60)
        print(
            f"{'Mean Target Similarity':<30} {vl_valid['target_sim'].mean():>15.4f} {lm_valid['target_sim'].mean():>15.4f}"
        )
        print(
            f"{'Std Target Similarity':<30} {vl_valid['target_sim'].std():>15.4f} {lm_valid['target_sim'].std():>15.4f}"
        )

        if "target_dist" in vl_valid.columns:
            print(
                f"{'Mean Target Distance':<30} {vl_valid['target_dist'].mean():>15.4f} {lm_valid['target_dist'].mean():>15.4f}"
            )

        if "nearest_is_target" in vl_valid.columns:
            print(
                f"{'Nearest is Target %':<30} {vl_valid['nearest_is_target'].mean() * 100:>15.1f} {lm_valid['nearest_is_target'].mean() * 100:>15.1f}"
            )

    # Save detailed results
    vl_results.to_csv("hypothesis_test_vl_results.csv", index=False)
    lm_results.to_csv("hypothesis_test_lm_results.csv", index=False)
    print(
        "\nDetailed results saved to hypothesis_test_vl_results.csv and hypothesis_test_lm_results.csv"
    )

    # Visualization
    print("\nGenerating visualizations...")
    visualize_examples(
        vl_results, meta, n_examples=16, filename="hypothesis_visualization_vl.pdf"
    )
    visualize_examples(
        lm_results, meta, n_examples=16, filename="hypothesis_visualization_lm.pdf"
    )


if __name__ == "__main__":
    main()
