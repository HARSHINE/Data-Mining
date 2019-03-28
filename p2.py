# References: https://sebastianraschka.com/blog/index.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity

import prettytable
import numpy as np
import matplotlib.pyplot as plt

XInside = np.array([[0, 0, 0], [0.2, 0.2, 0.2], [0.1, -0.1, -0.3]])

XOutside = np.array([[-1.2, 0.3, -0.3], [0.8, -0.82, -0.9], [1, 0.6, -0.7],
                     [0.8, 0.7, 0.2], [0.7, -0.8, -0.45], [-0.3, 0.6, 0.9],
                     [0.7, -0.6, -0.8]])


def hypercubeKernel(h, x, x_i):
    assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
    return (x - x_i) / (h)


def parzenWindowFn(x_vec, h=0.1):
    for row in x_vec:
        if np.abs(row) > (1 / 2):
            return 0
    return 1


X = np.vstack((XInside, XOutside))
assert (X.shape == (10, 3))

k_n = 0
for row in X:
    k_n += parzenWindowFn(row.reshape(3, 1))


def parzenEstimation(x_samples, point_x, h, d, window_func, kernel_func):
    k_n = 0
    for row in x_samples:
        x_i = kernel_func(h=h, x=point_x, x_i=row[:, np.newaxis])
        k_n += window_func(x_i, h=h)
    return (k_n / len(x_samples)) / (h ** d)


point_x = np.array([[0], [0], [0]])

# Generate 1000 random 2D-patterns
mu_vec = np.array([1, 0])
cov_mat = np.array([[0.9, 0.4], [0.4, 0.9]])
data1 = np.random.multivariate_normal(mu_vec, cov_mat, 500)
mu_vec2 = np.array([0, 1.5])
cov_mat2 = np.array([[0.9, 0.4], [0.4, 0.9]])
data2 = np.random.multivariate_normal(mu_vec2, cov_mat2, 500)
x_2Dgauss = np.concatenate((data1, data2))
'''mu = 5
sig = 6
x_2Dgauss = np.random.multivariate_normal(mu, sig, 1000)
'''
# import numpy as np
mu, sigma = 5, 1
# mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=10, color='r')
plt.title('Problem 2 Question 2')
plt.show()
mu2, sigma2 = 0, 0.2
# mean and standard deviation
s2 = np.random.normal(mu, sigma, 1000)
sfinal = np.concatenate((s, s2))
count2, bins2, ignored2 = plt.hist(sfinal, 30, normed=True)
plt.plot(bins2, 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu2) ** 2 / (2 * sigma2 ** 2)), linewidth=10,
         color='k')
plt.title('Problem 2 Question 3')
plt.show()


def mykde(x, mu, cov):
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))


p1 = parzenEstimation(x_2Dgauss, np.array([[0], [0]]), h=10, d=2,
                      window_func=parzenWindowFn,
                      kernel_func=hypercubeKernel)
p2 = parzenEstimation(x_2Dgauss, np.array([[0.5], [0.5]]), h=10, d=2,
                      window_func=parzenWindowFn,
                      kernel_func=hypercubeKernel)
p3 = parzenEstimation(x_2Dgauss, np.array([[0.3], [0.2]]), h=10, d=2,
                      window_func=parzenWindowFn,
                      kernel_func=hypercubeKernel)

mu = np.array([[0], [0]])
cov = np.eye(2)

a1 = mykde(np.array([[0], [0]]), mu, cov)
a2 = mykde(np.array([[0.5], [0.5]]), mu, cov)
a3 = mykde(np.array([[0.3], [0.2]]), mu, cov)

results = prettytable.PrettyTable(["", "p(x) actual"])
results.add_row(["p([0,0]^t", a1])
results.add_row(["p([0.5,0.5]^t", a2])
results.add_row(["p([0.3,0.2]^t", a3])

print(results)
