import sys
import numpy as np
from scipy.io import wavfile
import scipy.io

cost = 0


# this function finds the index of the closest centroid to a point
def find_closest(p):
    closest = 0
    # the minimum distance
    dist = sys.maxsize
    for cent in range(k):
        d = np.linalg.norm(p - centroids[cent])
        # when 2 centroids are evenly close to a certain point, the one with the lower index ”wins”
        if d < dist:
            dist = d
            closest = cent
    # evaluate loss cost
    global cost
    cost += pow(dist, 2)
    return closest


def update_centroids(clusters):
    for i in range(k):
        # if there is no points assigned to current i centroid don't change it
        if not clusters[i]:
            continue
        # update the centroid i by the average formula
        sigma = np.average(clusters[i], 0)
        centroids[i][0], centroids[i][1] = np.round(sigma[0]), np.round(sigma[1])


sample, centroidsF = sys.argv[1], sys.argv[2]
rate, y = scipy.io.wavfile.read(sample, mmap=False)
# create a copy from points array to replace each value by its centroid
x = np.array(y.copy())
centroids = np.loadtxt(centroidsF)
k = int(centroids.size / 2)
num_of_points = int(y.size / 2)
f = open("output.txt", "w")
# run for 30 iterations or until convergence
for iteration in range(30):
    cost = 0
    # assign each point to the closest centroid
    assign = []
    for a in range(k):
        assign.append([])
    for point in range(num_of_points):
        centroid = find_closest(y[point])
        assign[centroid].append(y[point])
        # replace each value by its centroid
        x[point] = centroids[centroid]

    # change centroids and check convergence
    array = np.array(assign, dtype=object)
    oldCentroids = np.copy(centroids)
    update_centroids(array)
    f.write(f"[iter {iteration}]:{','.join([str(i) for i in centroids])}")
    f.write("\n")
    # convergence
    if np.array_equal(oldCentroids, centroids):
        break
f.close()
scipy.io.wavfile.write("compressed.wav", rate, np.array(x, dtype=np.int16))
