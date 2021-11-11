import numpy as np

from scipy.stats import pearsonr, norm, rankdata
from scipy.spatial import KDTree
from scipy.special import digamma

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class ConditionalIndependenceTest:

    def __init__(self, significance_level=0.05, k_perm=5, bootstrap_surrogates=50):
        self.significance_level = significance_level
        self.k_perm = k_perm
        self.bootstrap_surrogates = bootstrap_surrogates

    def partial_correlation(self, x, y, z):

        x_standardized = StandardScaler().fit_transform(x)
        y_standardized = StandardScaler().fit_transform(y)
        z_standardized = StandardScaler().fit_transform(z)

        x_z = LinearRegression()
        x_z.fit(z_standardized, x_standardized)
        residual_x = (x_standardized - x_z.predict(z_standardized))

        y_z = LinearRegression()
        y_z.fit(z_standardized, y_standardized)
        residual_y = (y_standardized - y_z.predict(z_standardized))

        r, _ = pearsonr(residual_x.flatten(), residual_y.flatten())

        if np.sqrt(x.shape[0] - z.shape[1] - 3) * np.abs(0.5 * np.log((1.0 + r) / (1.0 - r))) > \
                norm.ppf(1.0 - self.significance_level / 2.0):
            return False
        else:
            return True

    def _add_noise(self, x, intens=1e-6):
        return x + intens * np.random.random_sample(x.shape)

    def _get_nearest_number_of_points(self, points, dvec):
        tree = KDTree(points)
        num_points = [len(elem) for elem in tree.query_ball_point(points, dvec)]
        return num_points

    def _conditional_mutual_information_calculation(self, x, y, z, k=10):
        points = np.concatenate([x, y, z], axis=1)

        tree = KDTree(points)
        dvec = tree.query(points, k=k + 1)[0][:, k]

        xz = np.concatenate([x, z], axis=1)
        k_xz = self._get_nearest_number_of_points(xz, dvec)
        yz = np.concatenate([y, z], axis=1)
        k_yz = self._get_nearest_number_of_points(yz, dvec)
        k_z = self._get_nearest_number_of_points(z, dvec)

        cmi = np.mean(digamma(k) - digamma(k_xz) - digamma(k_yz) + digamma(k_z))

        return cmi

    def nearest_neighbour_conditional_independence(self, x, y, z):

        x_rank = rankdata(self._add_noise(x), axis=0)
        y_rank = rankdata(self._add_noise(y), axis=0)
        z_rank = rankdata(self._add_noise(z), axis=0)

        initial_cmi = self._conditional_mutual_information_calculation(x_rank, y_rank, z_rank)

        nearest_neighbor_lookup = KDTree(z)
        nearest_neighbor_indices = {}
        for idx in range(x.shape[0]):
            _, nearest_neighbors = nearest_neighbor_lookup.query(z[idx, :].reshape(1, -1), self.k_perm)
            nearest_neighbor_indices[idx] = nearest_neighbors.flatten()

        conditional_mutual_informations = []
        for _ in range(self.bootstrap_surrogates):
            used_indices = set()
            x_asterisk = [None] * x_rank.shape[0]
            nearest_neighbor_indices_permutation = {idx: np.random.permutation(nearest_neighbor_indices[idx])
                                                    for idx in nearest_neighbor_indices}

            sample_permutation = np.random.permutation(range(x_rank.shape[0]))
            for i in sample_permutation:
                j = nearest_neighbor_indices_permutation[i][0]
                m = 0
                while j in used_indices and m < self.k_perm - 1:
                    m += 1
                    j = nearest_neighbor_indices_permutation[i][m]
                x_asterisk[i] = x_rank[j]
                used_indices.add(j)

            x_asterisk = np.array(x_asterisk).reshape(-1, 1)
            conditional_mutual_information =\
                self._conditional_mutual_information_calculation(x_asterisk, y_rank, z_rank)
            conditional_mutual_informations.append(conditional_mutual_information)

        p = sum([bootstrap_cmi >= initial_cmi
                 for bootstrap_cmi in conditional_mutual_informations]) / self.bootstrap_surrogates
        if p < self.significance_level:
            return False
        else:
            return True
