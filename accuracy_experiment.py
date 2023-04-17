from ALoKDE.alokde_1d import ALoKDE, norm
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from os import mkdir

np.random.seed(42)
np.set_printoptions(precision=6)


def compute_l2_error(alg: ALoKDE, mu: float) -> float:
    n_points: int = 1001
    domain: Tuple[float, float] = alg.get_domain()

    step_size: float = (domain[1] - domain[0]) / n_points

    x: float = domain[0]

    est_y: List[float] = []
    the_y: List[float] = []

    while x < domain[1]:
        est_y.append(alg.pdf(x))
        the_y.append(norm.pdf(x, loc=mu, scale=1))
        x += step_size

    l2: float = 0

    for i in range(len(est_y)):
        l2 += (est_y[i] - the_y[i]) ** 2

    return np.sqrt(l2 * step_size)


def main():

    X = [i / 100 for i in range(-500, 4800)]

    theoretical_means = [i / 1000 for i in range(2000)]
    theoretical_means.extend([2 + i / 100 for i in range(4000)])
    theoretical_means.extend([42 for i in range(2000)])
    theoretical_means.extend([43 for i in range(2000)])

    h_mod: float = 2.7
    alpha: float = 0.05
    tau: float = 1
    e: float = 0.1
    m_t: int = 100

    alg = ALoKDE(e=e, tau=tau, alpha=alpha, h_mod=h_mod, m_t=m_t)

    dir: str = f"ALoKDE_alpha={alpha}_e={e}_tau={tau}_hmod={h_mod}_mt={m_t}"
    mkdir(dir)

    stream = []

    with open("stream.txt", "r") as f:
        while True:
            try:
                stream.append(float(f.readline()))
            except Exception:
                break

    sum_l2: float = 0

    for i, s in enumerate(stream):

        print(f"step {i+1}")

        alg.process_new_element(s)

        if (i + 1) % 10 == 0:
            updated_alokde_pdf = [alg.pdf(x) for x in X]
            norm_pdf = [norm.pdf(x, loc=theoretical_means[i], scale=1) for x in X]

            l2: float = compute_l2_error(alg, theoretical_means[i])
            sum_l2 += l2

            plt.text(-4.5, 0.52, f"i = {i + 1}, "
                                 f"l2_a = {round(l2, 6)}, "
                                 f"avg_l2 = {round(sum_l2 / ((i+1) / 10), 6)}")

            plt.xlim(X[0], X[-1])
            plt.ylim(0, 0.5)
            plt.plot([s], [0], marker="x", ms=10)
            plt.plot(X, norm_pdf, color="red")
            plt.plot(X, updated_alokde_pdf, color="black")
            #plt.show()
            plt.savefig(f"{dir}/{i+1}.png")
            plt.clf()

    print("Done")

if __name__=="__main__":
    main()
