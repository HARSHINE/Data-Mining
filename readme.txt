
HARSHINE VENKATARAMAN
1001569298
CSE5334
Homework-1
March 10,2019

README:

PROBLEM1)

PRE-REQUISITE: PYCHARM IDE:
FILENAME:  p1.py
Open P1.py and click run

PROBLEM2)
PRE-REQUISITE: PYCHARM IDE:
FILENAME:  p2.py
Open P2.py and click run

QUESTION2) to change h value: change linewidth value here:
line#63: plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=10, color='r')

QUESTION3)to change h value: change linewidth value here:
line#71: plt.plot(bins2, 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu2) ** 2 / (2 * sigma2 ** 2)), linewidth=10,
         color='k')
Question4)to change h value:

line#83: p1 = parzenEstimation(x_2Dgauss, np.array([[0], [0]]), h=10, d=2,
                      window_func=parzenWindowFn,
                      kernel_func=hypercubeKernel)
line#86: p2 = parzenEstimation(x_2Dgauss, np.array([[0.5], [0.5]]), h=10, d=2,
                      window_func=parzenWindowFn,
                      kernel_func=hypercubeKernel)
line#89: p3 = parzenEstimation(x_2Dgauss, np.array([[0.3], [0.2]]), h=10, d=2,
                      window_func=parzenWindowFn,
                      kernel_func=hypercubeKernel)




