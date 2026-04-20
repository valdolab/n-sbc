"""Plotting utilities for n-SBC explainability."""

import numpy as np


def _get_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install nsbc[viz]"
        ) from err


def plot_feature_importances(z_matrix, sample_idx=None, feature_names=None, ax=None):
    """Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    z_matrix : ZMatrix
    sample_idx : int or None
        If None, plot global importances. If int, plot local for that sample.
    feature_names : list of str or None
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    plt = _get_matplotlib()

    if sample_idx is not None:
        importances = z_matrix.feature_importances[sample_idx]
        title = f"Feature importances (sample {sample_idx})"
    else:
        importances = z_matrix.global_feature_importances
        title = "Global feature importances"

    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, n_features * 0.4)))
    else:
        fig = ax.figure

    y_pos = np.arange(n_features)
    ax.barh(y_pos, importances, edgecolor="k", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Match ratio")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    return fig, ax


def plot_similarity_heatmap(
    z_matrix, sample_idx, feature_names=None, top_k=None, ax=None
):
    """Plot per-feature match ratios between a test sample and its top-u neighbors.

    Parameters
    ----------
    z_matrix : ZMatrix
    sample_idx : int
    feature_names : list of str or None
    top_k : int or None
        Number of training samples to show. Defaults to n_value.
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    plt = _get_matplotlib()
    from nsbc.tools.matrix_z import compute_feature_match_ratios

    pred_label = z_matrix.predictions[sample_idx]
    top_indices = z_matrix.top_u_indices[sample_idx][pred_label]
    if top_k is not None:
        top_indices = top_indices[:top_k]

    n_features = len(z_matrix.feature_bit_widths)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    z_scores = z_matrix.z[sample_idx, top_indices]
    ratios = compute_feature_match_ratios(
        z_matrix.x_test_encoded[sample_idx],
        z_matrix.x_train_encoded[top_indices],
        z_matrix.feature_bit_widths,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, n_features * 0.8), max(3, len(top_indices) * 0.6))
        )
    else:
        fig = ax.figure

    train_labels = [
        f"Train {idx} (z={z_scores[j]})" for j, idx in enumerate(top_indices)
    ]

    im = ax.imshow(ratios, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(train_labels)
    ax.set_title(f"Feature match ratios (sample {sample_idx}, pred={pred_label})")
    fig.colorbar(im, ax=ax, label="Match ratio")
    return fig, ax


def plot_chunk_similarity(
    z_matrix, sample_idx, y_train, top_k=10, feature_names=None, ax=None
):
    """Two-panel plot: chunked bit-match ratios (left) + z-scores with arrows (right).

    Parameters
    ----------
    z_matrix : ZMatrix
    sample_idx : int
    y_train : array-like
        Training labels for grouping by class.
    top_k : int
        Number of top training samples to show per class.
    feature_names : list of str or None
        Labels for the feature chunks on the x-axis.
    ax : tuple of (ax1, ax2) or None

    Returns
    -------
    fig, (ax1, ax2)
    """
    plt = _get_matplotlib()
    from matplotlib import cm

    from nsbc.tools.matrix_z import compute_feature_match_ratios

    y_train = np.asarray(y_train)
    z_vec = z_matrix.z[sample_idx]
    class_list = list(z_matrix.classes)
    n_value = z_matrix.n_value
    feature_bit_widths = z_matrix.feature_bit_widths
    n_features = len(feature_bit_widths)

    display_indices = []
    for cls in class_list:
        cls_mask = np.where(y_train == cls)[0]
        cls_z = z_vec[cls_mask]
        order = np.argsort(-cls_z)
        k = min(top_k, len(order))
        display_indices.append(cls_mask[order[:k]])
    display_indices = np.concatenate(display_indices)
    n_shown = len(display_indices)

    # Compute per-feature match ratios for displayed samples
    ratios = compute_feature_match_ratios(
        z_matrix.x_test_encoded[sample_idx],
        z_matrix.x_train_encoded[display_indices],
        feature_bit_widths,
    )

    # Bit boundaries for chunk widths
    ends = np.cumsum(feature_bit_widths)
    starts = np.concatenate([[0], ends[:-1]])
    total_bits = int(ends[-1])

    cmaps = [cm.Blues, cm.Oranges, cm.Greens, cm.Reds, cm.Purples]
    color_map = {cls: cmaps[i % len(cmaps)] for i, cls in enumerate(class_list)}
    bar_color = {cls: f"C{i}" for i, cls in enumerate(class_list)}

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(14, max(4, n_shown * 0.4)),
            gridspec_kw={"width_ratios": [4, 1]},
        )
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    # Left panel: chunked bars
    for i in range(n_shown):
        train_idx = display_indices[i]
        cls = y_train[train_idx]
        cmap_fn = color_map.get(cls, cm.Blues)
        left = 0
        for j in range(n_features):
            width = int(feature_bit_widths[j])
            ax1.barh(
                i,
                width=width,
                left=left,
                color=cmap_fn(ratios[i, j]),
                edgecolor="k",
                height=0.6,
            )
            left += width
        ax1.text(
            -total_bits * 0.02,
            i,
            f"{int(cls)}_{train_idx}",
            va="center",
            ha="right",
            fontsize=9,
        )
        ax1.text(
            left + total_bits * 0.01,
            i,
            f"z={z_vec[train_idx]}",
            va="center",
            fontsize=9,
        )

    ax1.set_yticks([])
    if feature_names is None:
        xlabels = [f"{starts[j]}-{ends[j] - 1}" for j in range(n_features)]
    else:
        xlabels = feature_names
    ax1.set_xticks(starts)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")
    ax1.set_xlim(-total_bits * 0.05, total_bits)
    ax1.set_xlabel("Bits grouped by feature")
    ax1.set_title(f"Match ratios by feature chunk (sample {sample_idx})")

    # Right panel: z-score bars with arrows on top-n
    z_shown = z_vec[display_indices]
    colors = [bar_color.get(y_train[idx], "C0") for idx in display_indices]
    ax2.barh(range(n_shown), z_shown, color=colors, edgecolor="k", height=0.6)
    ax2.set_yticks([])
    ax2.set_xlim(0, total_bits)
    ax2.set_xlabel("Score z")
    ax2.set_title("Total z")

    # Arrows on top-n_value per class
    for cls in class_list:
        cls_mask = np.where(y_train[display_indices] == cls)[0]
        cls_z = z_shown[cls_mask]
        top_n = min(n_value, len(cls_z))
        top_rows = cls_mask[np.argsort(-cls_z)[:top_n]]
        for r in top_rows:
            ax2.annotate(
                "",
                xy=(z_shown[r], r),
                xytext=(z_shown[r] + total_bits * 0.1, r),
                arrowprops=dict(arrowstyle="->", color="red", lw=2, mutation_scale=20),
            )

    return fig, (ax1, ax2)


def plot_z_scores(z_matrix, sample_idx, y_train=None, top_k=10, ax=None):
    """Plot top-k Z-scores per class, grouped and sorted by similarity.

    Parameters
    ----------
    z_matrix : ZMatrix
    sample_idx : int
    y_train : array-like or None
        Training labels for coloring and grouping bars by class.
    top_k : int or None
        Number of top training samples to show per class. None shows all.
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    plt = _get_matplotlib()

    z_vec = z_matrix.z[sample_idx]
    pred_label = z_matrix.predictions[sample_idx]
    n_bits = z_matrix.x_test_encoded.shape[1]
    class_list = list(z_matrix.classes)
    color_map = {cls: f"C{i}" for i, cls in enumerate(class_list)}

    if y_train is not None:
        y_train = np.asarray(y_train)
        display_indices = []
        for cls in class_list:
            cls_mask = np.where(y_train == cls)[0]
            cls_z = z_vec[cls_mask]
            sorted_order = np.argsort(-cls_z)
            k = len(sorted_order) if top_k is None else min(top_k, len(sorted_order))
            display_indices.append(cls_mask[sorted_order[:k]])
        display_indices = np.concatenate(display_indices)
    else:
        sorted_order = np.argsort(-z_vec)
        k = (
            len(sorted_order)
            if top_k is None
            else min(top_k * len(class_list), len(sorted_order))
        )
        display_indices = sorted_order[:k]

    n_shown = len(display_indices)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, n_shown * 0.35)))
    else:
        fig = ax.figure

    z_shown = z_vec[display_indices]
    colors = (
        [color_map.get(y_train[i], "C0") for i in display_indices]
        if y_train is not None
        else "steelblue"
    )

    ax.barh(range(n_shown), z_shown, color=colors, edgecolor="k", height=0.6)

    for row, idx in enumerate(display_indices):
        label = f"{idx}" if y_train is None else f"{int(y_train[idx])}_{idx}"
        ax.text(-n_bits * 0.02, row, label, va="center", ha="right", fontsize=9)
        ax.text(
            z_vec[idx] + n_bits * 0.02,
            row,
            f"{z_vec[idx]}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks([])
    ax.set_xlim(-n_bits * 0.15, n_bits * 1.15)
    ax.set_xlabel("Hamming similarity (z)")
    ax.set_title(
        f"Z-scores for sample {sample_idx} (pred={pred_label}), n={z_matrix.n_value}"
    )
    return fig, ax
