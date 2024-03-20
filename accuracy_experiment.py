from ALoKDE.alokde_1d import ALoKDE, norm
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from os import mkdir, walk, path
import time

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


def unimodal_pdf(x, mean: float) -> float:
    return norm.pdf(x, loc=mean, scale=1)

def bimodal_pdf(x, mean: float) -> float:
    return 0.6 * norm.pdf(x, loc=mean, scale=1) + 0.4 * norm.pdf(x, loc=mean + 5, scale=1)

def trimodal_pdf(x, mean: float) -> float:
    return 0.3 * norm.pdf(x, loc=mean -5, scale=1) + 0.3 * norm.pdf(x, loc=mean + 5, scale=1) + 0.4 * norm.pdf(x, loc=mean, scale=1)


def process_stream(stream_id, stream_num):

    start_time = time.time()

    X = [i / 100 for i in range(-500, 4800)]  # Unimodal
    if stream_id == 13:
        X = [i / 100 for i in range(-500, 5300)]  # Bimodal
    if stream_id == 14:
         X = [i / 100 for i in range(-1000, 5300)]  # Trimodal

    theoretical_means = [i / 1000 for i in range(2000)]
    theoretical_means.extend([2 + i / 100 for i in range(4000)])
    theoretical_means.extend([42 for i in range(2000)])
    theoretical_means.extend([43 for i in range(2000)])

    h_mod: float = 1        # Default = 1
    alpha: float = 0.05     # Default = 0.05
    tau: float = 1          # Default = 1
    e: float = 0.1          # Default = 0.1
    m_t: int = 100          # Default -> Not discussed in the paper.

    alg = ALoKDE(e=e, tau=tau, alpha=alpha, h_mod=h_mod, m_t=m_t)

    dir: str = f"y:/TR Badania/ALoKDE/results_{stream_id}"

    # Make dir if it doesn't exist.
    if not path.exists(dir):
        mkdir(dir)

    error_and_drawing_frequency: int = 10
    results_file: str = f"results_{stream_num}.csv"
    n_errors: int = 0
    sum_l2: float = 0

    # Check if file exists
    if results_file in [f for f in next(walk(dir))[2]]:
        with open(f"{dir}/{results_file}", "r") as f:
            lines = f.readlines()
            n_errors = len(lines)
            sum_l2 = float(lines[-1]) * n_errors

    if n_errors >= 1000:
        print(f"Stream {stream_num} already processed. Skipping...")
        return

    error_renew_start: int = n_errors * error_and_drawing_frequency

    stream = []

    # stream_file = f"stream_{stream_id}/stream_{stream_id}_{stream_num}.csv"
    stream_file = f"y:/data/stream_{stream_id}/stream_{stream_id}_{stream_num}.csv"

    with open(stream_file, "r") as f:
        while True:
            try:
                stream.append(float(f.readline()))
            except Exception:
                break

    for i, s in enumerate(stream):

        print(f"step {i+1}")

        alg.process_new_element(s)

        if (i + 1) % error_and_drawing_frequency == 0 and (i + 1) > error_renew_start:

            print("Computing error...")

            updated_alokde_pdf = [alg.pdf(x) for x in X]
            #norm_pdf = [unimodal_pdf(x, theoretical_means[i]) for x in X]
            norm_pdf = [bimodal_pdf(x, theoretical_means[i]) for x in X]
            # norm_pdf = [trimodal_pdf(x, theoretical_means[i]) for x in X]

            l2: float = compute_l2_error(alg, theoretical_means[i])
            sum_l2 += l2
            avg_l2 = round(sum_l2 / ((i+1) / error_and_drawing_frequency), 6)

            with open(f"{dir}/{results_file}", "a") as f:
                f.write(f"{avg_l2}\n")

            print("Drawing...")

            # Plot if it's the first stream of the list.
            if stream_num < 5:
                plt.text(-4.5, 0.52, f"i = {i + 1}, "
                                     f"l2_a = {round(l2, 6)}, "
                                     f"avg_l2 = {avg_l2}")

                plt.xlim(X[0], X[-1])
                plt.ylim(-0.1, 0.5)
                plt.plot([s], [0], marker="x", ms=10)
                plt.plot(X, norm_pdf, color="red")
                plt.plot(X, updated_alokde_pdf, color="black")
                #plt.show()
                plt.savefig(f"{dir}/{stream_num}_{i+1}.png")
                plt.clf()

    print(f"Done. The experiment took {time.time() - start_time}.")


def main():
    stream_id = 13
    end_seed = 20

    for stream_number in range(end_seed - 19, end_seed + 1):
        process_stream(stream_id, stream_number)


if __name__=="__main__":
    main()
