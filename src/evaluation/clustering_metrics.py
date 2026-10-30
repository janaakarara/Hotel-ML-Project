"""Internal clustering scores: they look only at the prepared data and the cluster labels, never at an outcome."""
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def score_clustering(X, labels, silhouette_sample, seed):
    return {
        # -1 to 1, higher is better: each point is close to its own cluster and far from the nearest other one.
        # It compares every pair of points, so we score a fixed random sample to keep it fast.
        "silhouette": silhouette_score(X, labels, sample_size=min(silhouette_sample, len(X)), random_state=seed),
        # 0 and up, lower is better: how much each cluster overlaps with the cluster most similar to it.
        "davies_bouldin": davies_bouldin_score(X, labels),
        # 0 and up, higher is better: spread between clusters divided by spread inside clusters.
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }
