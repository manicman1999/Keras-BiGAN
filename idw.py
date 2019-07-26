import numpy as np
import matplotlib.pyplot as plt
import random
import time

def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2) + 1e-7

def IDW(location, points, values, exp = 35, dim_weights = 1):
    start = time.clock()
    value = 0
    weight = 0

    for p in range(len(values)):
        w = get_distance(location * dim_weights, points[p] * dim_weights) ** exp
        v = values[p] / w

        weight = weight + (1/w)
        value = value + v

    #print(str(round(time.clock() - start, 5)) + " seconds on " + str(len(points)) + " samples.")

    return value / weight
