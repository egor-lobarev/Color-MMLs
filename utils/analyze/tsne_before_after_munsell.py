from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


@dataclass
class MunsellSubsetSpec:
    variables: list[str]
    values: list[list[int | str]]
    fixed_h: list[str | None]
    fixed_c: list[int | None]
    fixed_v: list[int | None]
    perplexity: int = 6


@dataclass
class TSNEBeforeAfterResult:
    selected_table: pd.DataFrame
    selected_indices: np.ndarray
    tsne_before: np.ndarray
    tsne_after: np.ndarray
    perplexity_used: float
    missing_specs: list[str]


def default_chroma_subset_spec(perplexity: int = 6) -> MunsellSubsetSpec:
    return MunsellSubsetSpec(
        variables=["c"] * 5,
        values=[
            [0, 2, 4, 6, 8],
            [0, 2, 4, 6, 8],
            [0, 2, 4, 6, 8, 10, 12, 14],
            [0, 2, 4, 6, 8, 10, 12, 14],
            [0, 2, 4, 6],
        ],
        fixed_h=["5Y", "5G", "2.5R", "5P", "5B"],
        fixed_c=[None] * 5,
        fixed_v=[5] * 5,
        perplexity=perplexity,
    )


def _normalize_hue(hue: str | float | int | None) -> str | None:
    if hue is None:
        return None
    if isinstance(hue, (float, int)):
        return f"{float(hue):.1f}R"

    hue = str(hue).strip().upper()
    if hue == "N":
        return "N"

    match = re.match(r"^(\d+(?:\.\d+)?)([A-Z]+)$", hue)
    if not match:
        return hue

    num, suffix = match.groups()
    value = float(num)
    if value.is_integer():
        num_fmt = f"{value:.1f}"
    else:
        num_fmt = f"{value:g}"
    return f"{num_fmt}{suffix}"


def _collect_requested_colors(
    manifest: pd.DataFrame,
    spec: MunsellSubsetSpec,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    df = manifest.copy()
    df["H_norm"] = df["H"].astype(str).map(_normalize_hue)
    df["picture_id"] = (
        df["picture"]
        .astype(str)
        .str.replace(".png", "", regex=False)
        .astype(int)
    )

    rows = []
    missing = []
    for chain_id, (variable, vals, h_fix, c_fix, v_fix) in enumerate(
        zip(spec.variables, spec.values, spec.fixed_h, spec.fixed_c, spec.fixed_v)
    ):
        variable = variable.lower()
        for order_in_chain, val in enumerate(vals):
            h_target = _normalize_hue(h_fix if variable != "h" else val)
            c_target = int(c_fix if variable != "c" else val)
            v_target = int(v_fix if variable != "v" else val)

            matched = df[
                (df["H_norm"] == h_target) &
                (df["C"].astype(int) == c_target) &
                (df["V"].astype(int) == v_target)
            ]
            if matched.empty:
                missing.append((h_target, v_target, c_target))
                continue

            row = matched.iloc[0].copy()
            row["manifest_row"] = int(matched.index[0])
            row["chain_id"] = chain_id
            row["order_in_chain"] = order_in_chain
            row["munsell_spec"] = f"{h_target} {v_target}/{c_target}"
            rows.append(row)

    missing_specs = [f"{h} {v}/{c}" for h, v, c in missing]
    if strict and missing_specs:
        missing_str = ", ".join(missing_specs[:10])
        suffix = "..." if len(missing_specs) > 10 else ""
        raise ValueError(f"Colors not found in manifest: {missing_str}{suffix}")

    selected = pd.DataFrame(rows).reset_index(drop=True)
    return selected, missing_specs


def _resolve_embedding_indices(
    selected: pd.DataFrame,
    embeddings: np.ndarray,
    manifest_len: int,
    embedding_ids: Iterable[int] | None = None,
) -> np.ndarray:
    if embedding_ids is not None:
        id_to_pos = {int(idx): pos for pos, idx in enumerate(embedding_ids)}
        indices = selected["picture_id"].map(id_to_pos)
        if indices.isna().any():
            missing_ids = selected.loc[indices.isna(), "picture_id"].tolist()
            raise ValueError(
                "Some selected picture ids are missing in embedding_ids: "
                f"{missing_ids[:10]}"
            )
        return indices.astype(int).to_numpy()

    if len(embeddings) == manifest_len:
        # Most common case: embeddings are aligned with CSV rows.
        return selected["manifest_row"].astype(int).to_numpy()

    # Fallback: embedding rows correspond to folder ids 1..N.
    picture_based = selected["picture_id"].to_numpy() - 1
    if picture_based.max() < len(embeddings) and picture_based.min() >= 0:
        return picture_based

    raise ValueError(
        "Cannot map selected colors to embedding rows automatically. "
        "Pass embedding_ids (folder/picture ids in the same order as embeddings)."
    )


def _fit_tsne(data: np.ndarray, perplexity: int, random_state: int = 42) -> tuple[np.ndarray, float]:
    n_samples = data.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 points.")

    used_perplexity = float(min(perplexity, max(2, n_samples - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=used_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
    )
    return tsne.fit_transform(data), used_perplexity


def tsne_before_after_on_munsell_subset(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    manifest_path: str | Path = "data/colors/munsell_colors/munsell_manifest.csv",
    spec: MunsellSubsetSpec | None = None,
    embedding_ids: Iterable[int] | None = None,
    strict: bool = False,
    random_state: int = 42,
) -> TSNEBeforeAfterResult:
    if embeddings_before.shape[0] != embeddings_after.shape[0]:
        raise ValueError(
            "embeddings_before and embeddings_after must have the same number of rows."
        )

    manifest = pd.read_csv(manifest_path)
    if spec is None:
        spec = default_chroma_subset_spec()

    selected, missing_specs = _collect_requested_colors(manifest, spec, strict=strict)
    if selected.empty:
        raise ValueError("No colors were selected from manifest for the provided specification.")

    selected_indices = _resolve_embedding_indices(
        selected=selected,
        embeddings=embeddings_before,
        manifest_len=len(manifest),
        embedding_ids=embedding_ids,
    )

    before_subset = embeddings_before[selected_indices]
    after_subset = embeddings_after[selected_indices]

    tsne_before, used_perplexity = _fit_tsne(before_subset, spec.perplexity, random_state)
    tsne_after, _ = _fit_tsne(after_subset, spec.perplexity, random_state)

    return TSNEBeforeAfterResult(
        selected_table=selected,
        selected_indices=selected_indices,
        tsne_before=tsne_before,
        tsne_after=tsne_after,
        perplexity_used=used_perplexity,
        missing_specs=missing_specs,
    )


def plot_tsne_before_after(
    result: TSNEBeforeAfterResult,
    title_prefix: str = "LM",
    point_size: int = 110,
    annotate: bool = True,
    save_name: str = None
) -> None:
    rgb = result.selected_table[["R", "G", "B"]].to_numpy(dtype=float)
    labels = result.selected_table["munsell_spec"].astype(str).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xy, title in [
        (axes[0], result.tsne_before, "Before training transform"),
        (axes[1], result.tsne_after, "After training transform"),
    ]:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=rgb,
            s=point_size,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.95,
        )
        if annotate:
            for i, label in enumerate(labels):
                ax.annotate(label, (xy[i, 0], xy[i, 1]), fontsize=8, alpha=0.8)
        ax.set_title(f"{title_prefix}: {title}")
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{title_prefix} | t-SNE (perplexity={result.perplexity_used:g})")
    plt.tight_layout()
    if save_name is not None:
        import os
        os.makedirs("graphics", exist_ok=True)
        path = os.path.join("graphics", f"{save_name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
