from ALoKDE.alokde_1d import ALoKDE, norm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = [i / 100 for i in range(-500, 4800)]

theoretical_means = [i / 1000 for i in range(2000)]
theoretical_means.extend([2 + i / 100 for i in range(4000)])
theoretical_means.extend([42 for i in range(2000)])
theoretical_means.extend([43 for i in range(2000)])


alg = ALoKDE()

# alokde_pdf =[alg.pdf(x) for x in X]

norm_pdf = [norm.pdf(x, loc=0, scale=1) for x in X]
plt.plot(X, norm_pdf, color="red")
# plt.plot(X, alokde_pdf, color="blue")

stream = []

with open("stream.txt", "r") as f:
    while True:
        try:
            stream.append(float(f.readline()))
        except Exception:
            break

updated_alokde_pdf = [alg.pdf(x) for x in X]

for i, s in enumerate(stream):

    print(f"step {i+1}")

    alg.process_new_element(s)

    if (i + 1) % 10 == 0:
        updated_alokde_pdf = [alg.pdf(x) for x in X]
        norm_pdf = [norm.pdf(x, loc=theoretical_means[i], scale=1) for x in X]

        plt.xlim(X[0], X[-1])
        plt.ylim(0, 0.5)
        plt.plot([s], [0], marker="x", ms=10)
        plt.plot(X, norm_pdf, color="red")
        plt.plot(X, updated_alokde_pdf, color="black")
        #plt.show()
        plt.savefig(f"results/{i+1}.png")
        plt.clf()


print("Done")
