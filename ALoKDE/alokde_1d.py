"""
    This module contains the simplified (restricted to 1D) implementation of the ALoKDE
    algorithm proposed in https://doi.org/10.1007/s13042-021-01275-y.

    Disclaimer. I'm not the author of the algorithm.
"""
__author__ = "Tomasz Rybotycki"

import numpy as np
from scipy.stats import norm
from typing import List, Tuple, Callable, Optional
import math

class ALoKDE():
    """
    An implementation of the 1D ALoKDE algorithm.
    """

    def __init__(self, e: float = 0.1, tau: float = 1, m_t: int = 100) -> None:
        """
        The constructor of our implementation of the ALoKDE algorithm. The initial
        values were proposed by the authors of the algorithm.

        :note:
            Notice that we use constant :math:`m_t` number of samples for the new
            estimators. That's because the authors of the original algorithm didn't
            elaborate on the number of samples they require, and we found that 1000
            samples is enough for the

        :param e:
            :math:`\epsilon` from the article. Used in computing the Mahalanobis
            distance.
        :param tau:
            :math:`\tau` from the article. Used to estimate the local samples'
            closeness.
        :param m_t:
           The number of local samples used to construct the estimator in step
           :math:`t`.
        """

        self.was_updated: bool = True

        self._e: float = e
        self._tau: float = tau
        self._m_t: int = m_t  # Selection based on our empirical investigations.

        # KS Test

        # 0.975 for alpha = 0.05, as suggested by the authors
        self._ks_test_critical_level: float = 0.975
        self._n_numerical_estimate_points: int = 1001

        # Estimator
        self._weights: List[float] = [1]
        self._hs: List[Tuple[float, float]] = [(1.0, 1.0)]

        # The authors suggest to initialize their algorithm with 0. To make the
        # initial estimator processable in the same manner as all the others, we
        # assume that there's an additional incoming sample with value 0.01.
        self._samples: List[List[float]] = [list(np.random.normal(0, size=1001))]

        # Constants
        self._u_k: float = 1  # Gaussian kernel constant
        self._w_k: float = 1 / (2 * np.sqrt(np.pi))  # Gaussian kernel constant

        # Custom
        self._weight_threshold: float = 0.01


    def pdf(self, x: float) -> float:
        """
        Computes the value of the ALoKDE estimator in the given point.

        :param x:
            Point in which PDF should be estimated.

        :return:
            ALoKDE estimator value in the given point.
        """
        val: float = 0

        for i in range(len(self._samples)):
            val += self._compute_estimator_value(x, i) * self._weights[i]

        return val

    def cdf(self, x: float) -> float:
        cdf: float = 0

        for i in range(len(self._samples)):
            samples: List[float] = self._samples[i]
            n = len(samples)

            for j in range(len(samples) - 1):
                cdf += norm.cdf(x, loc=samples[j], scale=self._hs[i][0]) * self._weights[i] / n

            cdf += norm.cdf(x, loc=samples[-1], scale=self._hs[i][-1]) * self._weights[i] / n

        return cdf

    def _should_update(self, x: float) -> bool:
        """
        Using one-sample KS test, decide if the ALoKDE estimator should be updated.
        :return:
        """

        statistics: float = 1 - self.cdf(x)
        statistics = max(1 - statistics, statistics)

        return statistics > self._ks_test_critical_level

    def _draw_local_sample(self, x: float) -> float:
        subestimator_index: int = np.random.choice(range(len(self._weights)), p=self._weights)

        h: float = self._hs[subestimator_index][0]
        mu: float = np.random.choice(self._samples[subestimator_index])

        low = norm.cdf(x-self._tau, loc=mu, scale=h)
        high = norm.cdf(x+self._tau, loc=mu, scale=h)

        u: float = np.random.uniform(low=low, high=high)

        sample: float = norm.ppf(u, loc=mu, scale=h)

        if sample == float("inf"):
            sample = x+self._tau

        if sample == float("-inf"):
            sample = x-self._tau

        return sample

    def _get_local_samples(self, x: float) -> None:
        """
        Using current estimator, generate :math:`m_t` new samples, close enough to
        :math:`x`. Add :math:`x` at the end of the samples list.

        :param x:
            Samples around which new samples should be generated.
        """

        local_samples: List[float] = []

        for _ in range(self._m_t):
            local_samples.append(self._draw_local_sample(x))

        local_samples.append(x)

        print(local_samples)

        self._samples.append(local_samples)

    def _compute_bandwidths(self, x: float) -> None:
        """
        Computes the bandwidths for "internal" and "external" parts of the new
        estimator.

        :param x:
            A new sample drawn from the analyzed data stream.
        """
        c: float = 1.06  # Gaussian kernel constant

        h_d: float = c * np.std(self._samples[-1][:self._m_t]) * pow(self._m_t, -0.2)

        print(f"c={c}\n"
              f"std={np.std(self._samples[-1][:self._m_t])}\n"
              f"mt^-0.2={pow(self._m_t, -0.2)}\n"
        )

        distances: List[float] = [
            self._distance(x, y) for y in self._samples[-1][:self._m_t]
        ]

        h_x: float = c * np.std(distances)

        print((h_d, h_x))

        self._hs.append((h_d, h_x))

    def _compute_plugin_c(self, xi: int, idx: int) -> float:
        c: float = np.math.factorial(xi)
        c /= np.math.factorial(xi // 2)
        c /= np.sqrt(np.pi)
        samples_std: float = np.std(self._samples[idx])
        c /= (2 * samples_std) ** (xi + 1)
        return c

    def _compute_plugin_capital_c(self, idx: int, h: float) -> float:
        c: float = 0
        samples: List[float] = self._samples[idx]
        m = len(samples)

        k_4 = lambda x: (x**4 - 6 * x**2 + 3) * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

        for si in samples:
            for sj in samples:
                c += k_4((si - sj) / h)

        c /= m ** 2 * h**5

        return c

    def _compute_mise(self, indices: List[int]) -> float:
        """
        Computes MISE of the estimator. We will use the formula from the Kulczycki's
        handbook.

        :return:
        """
        mise: float = 0

        for idx in indices:
            c: float = self._compute_plugin_c(6, idx)

            m: int = len(self._samples[idx])

            pow_arg: float = 2 * 3 / (np.sqrt(2 * np.pi) * self._u_k * m * c)
            h_i: float = pow(pow_arg, 1/7)
            zf:float = self._compute_plugin_capital_c(idx, h_i)

            h: float = (self._w_k / (zf * m * self._u_k ** 2)) ** 0.2

            mise += (h**4 * self._u_k**2 * zf / 4 + self._w_k / (m * h)) * self._weights[idx]

        return mise

    def _get_estimator_bias(self, e_indices: List[int], x: float) -> float:
        bias: float = 0

        for idx in e_indices:
            e_bias: float = 0.5 * self._u_k * self._weights[idx]
            e_bias *= \
                self._compute_estimator_value(x, idx, self.kernel_second_derivative)

            bias += e_bias

        return bias

    def _get_estimator_value(self, e_indices: List[int], x: float) -> float:
        value: float = 0

        for idx in e_indices:
            value += self._compute_estimator_value(x, idx) * self._weights[idx]

        return value

    def _get_estimators_expected_value(self, e_indices: List[int]) -> float:
        domain_range: Tuple[float, float] = self._find_estimator_domain(e_indices)
        domain: List[float] = [domain_range[0]]
        step_size: float = (domain_range[1] - domain_range[0]) / self._n_numerical_estimate_points

        while domain[-1] < domain_range[1]:
            domain.append(domain[-1] + step_size)

        y: List[float] = []

        for x in domain:
            y.append(self._get_estimator_value(e_indices, x))

        expected_value: float = 0

        for i in range(len(domain) - 1):
            expected_value += (y[i] + y[i + 1]) * step_size

        return expected_value

    def _compute_estimators_covariance(self) -> float:

        covariance: float = 0
        m: int = len(self._samples)

        new_samples: List[float] = self._samples[m-1]
        new_mean: float = np.mean(new_samples)

        for i, samples in enumerate(self._samples):

            sub_covariance: float = 0
            s_mean: float = np.mean(samples)

            for j in samples:

                for k in new_samples:

                    sub_covariance += (j - s_mean) * (k - new_mean)

            sub_covariance /= len(samples) * len(new_samples)


            covariance += sub_covariance * self._weights[i]


        return covariance



        # CODE BELOW IS NUMERICAL AND TAKES TOO LONG, SOMETHING IS WRONG

        domain_range: Tuple[float, float] = self._find_estimator_domain()
        step_size = (domain_range[1] - domain_range[0]) / self._n_numerical_estimate_points

        domain: List[float] = [domain_range[0]]

        while domain[-1] < domain_range[1]:
            domain.append(domain[-1] + step_size)

        y = []

        m: int = len(self._samples)

        for x in domain:

            f_t_bias: float = self._get_estimator_bias(list(range(m - 1)), x)
            f_kde_bias: float = self._get_estimator_bias([m - 1], x)

            f_t_value: float = self._get_estimator_value(list(range(m - 1)), x)
            f_kde_value: float = self._get_estimator_value([m - 1], x)

            f_t_expected_value: float = self._get_estimators_expected_value(list(range(m - 1)))
            f_kde_expected_value: float = self._get_estimators_expected_value([m - 1])

            estimators_covariance: float = f_t_value - f_t_expected_value
            estimators_covariance *= f_kde_value - f_kde_expected_value

            y.append(f_t_bias * f_kde_bias + estimators_covariance)

        covariance: float = 0

        for i in range(len(y) - 1):
            covariance += (y[i] + y[i + 1]) * step_size

        return covariance

    def _update_weights(self):
        """
        Updates weights of the estimators.

        :note:
            We will compute the estimate of the
            :math:`\int^{\infty}_{-\infty} |f"(x)|^2 dx`
            in the same way as it's done in the  plug-in bandwidth computation
            algorithm.

        :note:
            We will compute the MISE of the estimators (A and B values from the
            article) using the
        """

        # TODO: Lambdas
        n_subestimators: int = len(self._samples)

        a = self._compute_mise(list(range(n_subestimators - 1)))

        self._weights.append(1)
        b = self._compute_mise([n_subestimators - 1])

        c = self._compute_estimators_covariance()

        l = np.max([0, np.min([1.0, (b-c) / (a + b - 2 * c)])])

        print(f"lambda = {l}")

        for i, w in enumerate(self._weights):
            self._weights[i] = w * (1 - l)

        self._weights[-1] = l

    def _find_estimator_domain(self, e_indices: Optional[List[int]] = None) -> Tuple[float, float]:
        """
        Finds the domain of the estimator.

        :return:

        """

        if not e_indices:
            e_indices = list(range(len(self._samples)))

        h = max(self._hs[e_indices[0]])
        min_val: float = min(self._samples[e_indices[0]]) - 5 * h
        max_val: float = max(self._samples[e_indices[0]]) + 5 * h

        for i in e_indices:
            h = max(self._hs[i])
            min_val = min(min_val, min(self._samples[i]) - 5 * h)
            max_val = max(max_val, max(self._samples[i]) + 5 * h)

        return (min_val, max_val)

    def _normalize_weights(self) -> None:
        weights_sum: float = sum(self._weights)

        for i, w in enumerate(self._weights):
            self._weights[i] = w / weights_sum

    def _remove_old_estimators(self) -> None:
        while self._weights[0] < self._weight_threshold:
            self._weights.pop(0)
            self._samples.pop(0)
            self._hs.pop(0)
            self._normalize_weights()

    def process_new_element(self, x: float) -> None:
        """
        Process the new data stream element.
        :param x:
            New data stream element.
        """
        if not self._should_update(x):
            self.was_updated = False
            return

        print(f"Updating after receiving: {x}.")

        self.was_updated = True
        self._get_local_samples(x)
        self._compute_bandwidths(x)
        self._update_weights()
        self._remove_old_estimators()

    def _compute_estimator_value(self, x: float, e_i: int, K: Callable[[float], float] = None) -> float:
        """
        Computes the (sub)estimator value in :math:`x`.

        :param x:
            Point in which estimators' value is to be evaluated.
        :param e_i:
            Estimator index.

        :return:
            The value of the (sub)estimator in point :math:`x`.
        """
        if not K:
            K = self.kernel

        val: float = 0

        samples: List[float] = self._samples[e_i]
        hs: Tuple[float, float] = self._hs[e_i]

        for i in range(len(samples) - 1):
            s = samples[i]
            val += K((x - s) / hs[0]) / hs[0]

        val += K((x - samples[-1]) / hs[1]) / hs[1]

        return val / len(samples)

    def kernel(self, x: float) -> float:
        """
        Normal kernel.
        :param x:
            Argument of normal kernel.
        :return:
            Value of normal kernel in :math:`x`.
        """
        return 1 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2)

    def kernel_second_derivative(self, x: float) -> float:
        return (x**2 - 1) * self.kernel(x)

    def _distance(self, x1: float, x2: float) -> float:
        """
        Computes 1d Mahalanobis distance between given points.

        :param x1:
            First value.
        :param x2:
            Second value.
        :return:
            1d Mahalanobis distance, as presented in ALoKDE paper.
        """
        return np.sqrt((x1 - x2) * (1 + self._e)** -1 * (x1 - x2))

