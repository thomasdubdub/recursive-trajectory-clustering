import logging
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


class TrajRDBSCAN:
    def __init__(
        self,
        features,
        decrement_ratio=0.9,
        min_pts=10,
        n_iteration_limit=10,
        eps_factor=1.0,
    ):
        self.features = features
        self.decrement_ratio = decrement_ratio
        self.min_pts = min_pts
        self.n_iteration_limit = n_iteration_limit
        self.eps_factor = eps_factor
        self.anomalies_ = []

    def fit(self, X):
        self.anomalies_dict = defaultdict(list)
        self.cluster_index_lists = []
        self.labels_ = np.full(
            X.shape[0], -1, dtype=int
        )  # everything classified as anomaly by default
        initial_index_list = list(range(0, X.shape[0]))
        self._rdbscan(X, initial_index_list, 1)  # iteration 1

        for i in range(len(self.cluster_index_lists)):
            for index in self.cluster_index_lists[i]:
                self.labels_[index] = i

        for i in range(1, max(self.anomalies_dict, key=int) + 1):
            self.anomalies_.extend(self.anomalies_dict[i])

        print("Top 10 anomalies:")
        print(self.anomalies_[:10])

    def _rdbscan(self, X, index_list, n_iteration):
        if n_iteration <= self.n_iteration_limit:

            logging.info("iteration {}".format(n_iteration))
            logging.info("nb of samples {}".format(X.shape[0]))

            if X.shape[0] > self.min_pts:
                eps = (
                    self.eps_factor
                    * self._epsilon(X, self.min_pts)
                    * self.decrement_ratio ** (n_iteration - 1)
                )
                logging.info("epsilon {}".format(eps))
                clustering = DBSCAN(eps=eps, min_samples=self.min_pts).fit(X)
                logging.info(
                    "nb of anomalies {}".format(Counter(clustering.labels_)[-1])
                )

                if clustering.labels_.max() > 0:  # more than one cluster
                    for i in range(clustering.labels_.max() + 1):
                        indices = self._indices(clustering.labels_, i)
                        s_X = X[indices]
                        s_index_list = [index_list[k] for k in indices]
                        self._rdbscan(s_X, s_index_list, n_iteration + 1)
                    self._store_anomalies(
                        clustering.labels_, index_list, n_iteration
                    )
                else:
                    last_index_list = self._index_list(
                        clustering.labels_, clustering.labels_.max(), index_list
                    )  # max = 0 or -1
                    if len(last_index_list) > self.min_pts:
                        self.cluster_index_lists.append(last_index_list)
                    if clustering.labels_.max() == 0:  # store last anomalies
                        self._store_anomalies(
                            clustering.labels_, index_list, n_iteration
                        )
        else:
            logging.warning("max iteration reached")

    def _store_anomalies(self, labels, index_list, n_iteration):
        if Counter(labels)[-1] > 0:
            anomalies = self._index_list(labels, -1, index_list)
            self.anomalies_dict[n_iteration].extend(anomalies)

    def _index_list(self, labels, value, index_list):
        indices = self._indices(labels, value)
        return [index_list[i] for i in indices]

    def _indices(self, labels, value):
        return [i for i in range(labels.shape[0]) if labels[i] == value]

    def _epsilon(self, X, k):  # KNN elbow method
        NN = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = NN.kneighbors(X)
        distanceDec = sorted(distances[:, k - 1], reverse=True)
        kneedle = KneeLocator(
            indices[:, 0], distanceDec, curve="convex", direction="decreasing"
        )
        return distances[int(kneedle.knee), k - 1]