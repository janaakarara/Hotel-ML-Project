"""Command-line entry point.

Examples:
    python main.py cluster --algorithm kmeans
    python main.py cluster --algorithm kmeans --k 5
"""
import argparse

import pandas as pd

from src.config import load_config, project_path
from src.data.load import load_clean_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def cmd_cluster(df, config, algorithm, k):
    if algorithm == "kmeans":
        from src.clustering import kmeans_baseline
        scan, metrics, profile = kmeans_baseline.run(df, config, k)
    print("Task C - clustering (all cleaned rows, no outcome column used)\n")
    print("k scan:")
    print(scan.round(4).to_string(index=False))
    print(f"\nBaseline {algorithm} with k = {metrics['k']}:")
    for name in ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia", "fit_seconds", "pca_2d_variance_shown"]:
        print(f"  {name:22s} {metrics[name]:.4f}")
    print("\nCluster profiles:")
    print(profile.to_string())
    print(f"\nFigures in {config['paths']['figures']}/, tables in {config['paths']['tables']}/, "
          f"model in {config['paths']['artifacts']}/")


def main():
    parser = argparse.ArgumentParser(description="CSE382 City Hotel (H2) project")
    sub = parser.add_subparsers(dest="command", required=True)
    cluster = sub.add_parser("cluster", help="run the clustering analysis")
    cluster.add_argument("--algorithm", choices=["kmeans"], default="kmeans")
    cluster.add_argument("--k", type=int, default=None, help="force a number of clusters instead of the automatic choice")
    args = parser.parse_args()

    config = load_config()
    df = load_clean_data(project_path(config["paths"]["clean_data"]))
    if args.command == "cluster":
        cmd_cluster(df, config, args.algorithm, args.k)


if __name__ == "__main__":
    main()
