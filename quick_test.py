from ALoKDE.alokde_1d import ALoKDE, norm
import numpy as np
import matplotlib.pyplot as plt

X = [i / 100 for i in range(-500, 1000)]


alg = ALoKDE()

norm_pdf = [norm.pdf(x, loc=0, scale=1) for x in X]
alokde_pdf =[alg.pdf(x) for x in X]
plt.xlim(-5, 10)
plt.plot(X, norm_pdf)
plt.plot(X, alokde_pdf)

stream = np.random.normal(0, size=1000)

for s in stream:
    alg.process_new_element(s)

updated_alokde_pdf =[alg.pdf(x) for x in X]
plt.plot(X, updated_alokde_pdf)

plt.show()
print("Done")
