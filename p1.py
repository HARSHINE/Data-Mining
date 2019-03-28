#references: https://elearn.uta.edu/bbcswebdav/pid-8090434-dt-content-rid-139991592_2/courses/2192-DATA-MINING-25983-001/cse5334-s19-07_supervised_unsupervised_learning-2%282%29.pdf
# stats.stackexchange.com
import numpy as np

import matplotlib.pyplot as plt

'''data1 = np.random.multivariate_normal([1, 0], [[0.9, 0.4], [0.4, 0.9]], 500)
data2 = np.random.multivariate_normal([0, 1.5], [[0.9, 0.4], [0.4, 0.9]], 500)
X = np.array(data1, data2)
print(X)'''


def mykmeans(X, k, c):
    centroids = []

    # centroids = randomize_centroids(data, centroids, k)
    centroids = c
    # centroids.append(data[np.ndarray([10,10],[-10,-10])].flatten().tolist())
    oldCentroids = [[] for i in range(k)]

    iterations = 10000
    while not (hasConverged(centroids, oldCentroids, iterations)) and iterations != 0:
        iterations -= 1

        clusters = [[] for i in range(k)]

        #  assigning data points to clusters
        clusters = euclidean_dist(X, centroids, clusters)

        # recalculating the centroids
        index = 0
        for cluster in clusters:
            oldCentroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1

    print("TOTAL NO OF DATA INSTANCES: " + str(len(X)))
    print("TOTAL NUMBER OF ITERATIONS: " + str(iterations))
    print("NEW CENTROIDS OF EACH CLUSTER: " + str(centroids))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    centerColors = [colors[l] for l in range(k)]
    newCentroids = np.array(centroids)
    plt.scatter(newCentroids[:, 0], newCentroids[:, 1], marker='^', c='k')

    print("CLUSTERS:")
    j = 0
    for cluster in clusters:
        print("CLUSTER WITH SIZE " + str(len(cluster)) + " STARTS:")
        print(np.array(cluster).tolist())
        print("CLUSTER ENDS.")
        clt = np.array(cluster)
        plt.scatter(clt[:, 0], clt[:, 1], marker='o', c=centerColors[j])
        j += 1
    j = 0
    plt.show()
    return


# Calculating euclidean distance between a data point and all available cluster centroids

def euclidean_dist(X, centroids, clusters):
    for instance in X:

        mu_index = min([(i[0], np.linalg.norm(instance - centroids[i[0]])) \
                        for i in enumerate(centroids)], key=lambda t: t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    for cluster in clusters:
        if not cluster:
            cluster.append(X[np.random.randint(0, len(X), size=1)].flatten().tolist())

    return clusters


#  initial centroids-randomize
def randomize_centroids(X, centroids, k):
    for cluster in range(0, k):
        centroids.append(X[np.random.randint(0, len(X), size=1)].flatten().tolist())
    return centroids


# checking if clusters have converged
def hasConverged(centroids, oldCentroids, iterations):
    max_iterations = 10000
    if iterations > max_iterations:
        return True
    return oldCentroids == centroids


data1 = np.random.multivariate_normal([1, 0], [[0.9, 0.4], [0.4, 0.9]], 500)
data2 = np.random.multivariate_normal([0, 1.5], [[0.9, 0.4], [0.4, 0.9]], 500)
X = np.concatenate((data1, data2))

#mykmeans(X, 2, [[10, 10], [-10, -10]])
mykmeans(X, 4, [[10,10],[-10,-10],[10,-10],[-10,10]])

