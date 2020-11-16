import sys
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt

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
        centroids[i][0], centroids[i][1] = round(sigma[0]), round(sigma[1])


def analyze():
    average_cost_np = np.array(average_cost)
    iterations_np = np.array(iterations)
    # add title and axis names to show
    plt.title('K = ' + str(k))
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    # limits for the Y and X axis
    # ax = plt.gca()
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.ylim(0, np.amax(average_cost_np))
    plt.xlim(0, iterations_np.size)
    # The average cost value as a function of iterations
    plt.plot(iterations, average_cost_np, ':c')
    # show graphic
    plt.show()


sample, centroidsF = sys.argv[1], sys.argv[2]
rate, y = scipy.io.wavfile.read(sample, mmap=False)
# to report the k = 2, 4, 8, 16 cases of centroids
np.savetxt(centroidsF, np.random.randint(np.amin(y), np.amax(y), (2, 2)))
# create cost and iteration arrays to show report
average_cost = []
iterations = []
# create a copy from points array to replace each value by its centroid
x = np.array(y.copy())
centroids = np.loadtxt(centroidsF)
k = int(centroids.size / 2)
num_of_points = int(y.size / 2)
f = open("output.txt", "w")
# run for 30 iterations or until convergence
for iteration in range(10):
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
    # save the average cost per iteration and iteration for analyzing
    average_cost.append(round(cost / num_of_points))
    iterations.append(iteration)

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
# create graphs showing the average loss/cost value as a function of the iterations
analyze()
