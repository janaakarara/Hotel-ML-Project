"""Figures and tables that help us choose k and describe what each cluster is."""
import matplotlib

matplotlib.use("Agg")  # draw straight into files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.clustering_features import add_clustering_features

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#d9d8d3"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_k_scan(scan, chosen_k, path):
    panels = [("inertia", "Inertia (look for the elbow)"),
              ("silhouette", "Silhouette (higher is better)"),
              ("davies_bouldin", "Davies-Bouldin (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (column, title) in zip(axes, panels):
        _style(ax)
        ax.axvline(chosen_k, color=MUTED, linewidth=1)
        ax.plot(scan["k"], scan[column], color=BLUE, linewidth=2, marker="o", markersize=7)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("k (number of clusters)", color=MUTED)
        ax.set_xticks(scan["k"])
    axes[0].annotate(f"chosen k = {chosen_k}", (chosen_k, scan["inertia"].max()),
                     xytext=(6, 0), textcoords="offset points", color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca_2d(X, labels, centers, sample, seed, path):
    """One small panel per cluster: that cluster in blue on top of every other booking in gray."""
    pca = PCA(n_components=2, random_state=seed).fit(X)
    rows = np.random.default_rng(seed).choice(len(X), size=min(sample, len(X)), replace=False)
    points, sample_labels = pca.transform(X[rows]), labels[rows]
    center_points = pca.transform(centers)
    share = pca.explained_variance_ratio_

    k = len(centers)
    cols = min(k, 4)
    nrows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.6 * nrows), sharex=True, sharey=True, squeeze=False)
    for cluster, ax in enumerate(axes.flat):
        if cluster >= k:
            ax.set_visible(False)
            continue
        _style(ax)
        mine = sample_labels == cluster
        ax.scatter(points[~mine, 0], points[~mine, 1], s=4, color=GRAY, linewidths=0)
        ax.scatter(points[mine, 0], points[mine, 1], s=4, color=BLUE, linewidths=0)
        ax.scatter(*center_points[cluster], s=120, marker="X", color=INK, edgecolors="white", linewidths=1.5)
        ax.set_title(f"Cluster {cluster}  ({np.mean(labels == cluster):.0%} of bookings)", color=INK, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(f"PC1 ({share[0]:.0%} of variance)", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({share[1]:.0%} of variance)", color=MUTED)
    fig.suptitle(f"K-Means clusters projected to 2D with PCA (the two axes keep {share.sum():.0%} of the variance)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return share


def _top_value(values):
    shares = values.value_counts(normalize=True)
    return f"{shares.index[0]} ({shares.iloc[0]:.0%})"


def profile_clusters(df, labels):
    """Describe each cluster in booking terms.

    IsCanceled and ADR were NOT used to build the clusters. We only look at them
    afterwards, to see whether the segments differ in ways that matter to the hotel.
    """
    d = add_clustering_features(df)
    d["TotalNights"] = d["StaysInWeekendNights"] + d["StaysInWeekNights"]
    g = d.groupby(pd.Series(labels, index=d.index, name="cluster"))
    return pd.DataFrame({
        "bookings": g.size(),
        "share": (g.size() / len(d)).round(3),
        "lead_time_median": g["LeadTime"].median(),
        "nights_mean": g["TotalNights"].mean().round(2),
        "weekend_share_mean": g["WeekendShare"].mean().round(2),
        "adults_mean": g["Adults"].mean().round(2),
        "has_kids": g["HasKids"].mean().round(3),
        "repeated_guest": g["IsRepeatedGuest"].mean().round(3),
        "top_market_segment": g["MarketSegment"].agg(_top_value),
        "top_customer_type": g["CustomerType"].agg(_top_value),
        "top_meal": g["Meal"].agg(_top_value),
        "cancel_rate_not_used": g["IsCanceled"].mean().round(3),
        "adr_mean_not_used": g["ADR"].mean().round(1),
    })


def plot_profiles(profile, path):
    panels = [("lead_time_median", "Median lead time (days)", BLUE),
              ("nights_mean", "Mean nights", BLUE),
              ("weekend_share_mean", "Weekend share of nights", BLUE),
              ("adults_mean", "Mean adults per booking", BLUE),
              ("cancel_rate_not_used", "Cancel rate (NOT used to cluster)", ORANGE),
              ("adr_mean_not_used", "Mean ADR in EUR (NOT used to cluster)", ORANGE)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (column, title, color) in zip(axes.flat, panels):
        _style(ax)
        ax.bar(profile.index.astype(str), profile[column], color=color, width=0.7)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("cluster", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
