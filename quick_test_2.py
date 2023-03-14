from ALoKDE.alokde_1d import ALoKDE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm


X = [i / 100 for i in range(-500, 500)]
# y = [norm.pdf(x) for x in X]

# plt.plot(X, y)

mu = -2

x = 2.4536

print(f"{norm.ppf(norm.cdf(x, loc=mu), loc=mu)} == {x}")

x = 2.4536 - 0.1
print(f"{norm.ppf(norm.cdf(x, loc=mu), loc=mu)} == {x}")

x = 2.4536 + 0.1
print(f"{norm.ppf(norm.cdf(x, loc=mu), loc=mu)} == {x}")


alg = ALoKDE()
# y_alokde = [alg.compute_alokde_value(x) for x in X]

# plt.plot(X, y_alokde)

#norm_ppf = [norm.ppf(x, loc=0, scale=1) for x in X]
#alokde_ppf =[alg.ppf(x) for x in X]
#plt.plot(X, norm_ppf)
#plt.plot(X, alokde_ppf)

norm_cdf = [norm.cdf(x, loc=0, scale=1) for x in X]
alokade_cdf = [alg.cdf(x) for x in X]

plt.plot(X, norm_cdf)
plt.plot(X, alokade_cdf)

plt.show()
print("Done")
