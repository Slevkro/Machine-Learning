import numpy as np
import timeit

"""
Perhaps the most common summary statistics are the mean and standard deviation, 
which allow you to summarize the "typical" values in a dataset, but other 
aggregates are useful as well (the sum, product, median, minimum and maximum, 
quantiles, etc.)
"""

# Summing 

L = np.random.random(100)
print(sum(L))
print(np.sum(L))

"""
However, because it executes the operation in compiled code, 
NumPy's version of the operation is computed much more quickly
"""

# MINIMUM AND MAXIMUM
big_array = np.random.rand(1000000)
min(big_array), max(big_array)
np.min(big_array), np.max(big_array)

"""
Numpy version si faster but also works with python's version
"""

# WITH MULTIDIMENSIONAL ARRAYS

M = np.random.random((3, 4))
print(M)

M.min(axis=0)   # Min per column

M.max(axis=1)   # Max per row

np.max(M, axis = 1)

"""
Axis 0 refers to columns
Axis 1 refers to rows
"""

# PRACTICAL EXAMPLE USING PANDAS

import pandas as pd 
import matplotlib.pyplot as plt
#import seaborn; seaborn.set()  # set plot style

heights = np.array([189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193, 178, 173,
 174, 183, 183, 168, 170, 178, 182, 180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183,
 177, 185, 188, 188, 182, 185])

print("Mean height:       ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height:    ", heights.min())
print("Maximum height:    ", heights.max())

print("25th percentile:   ", np.percentile(heights, 25))
print("Median:            ", np.median(heights))
print("75th percentile:   ", np.percentile(heights, 75))

plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')
plt.show(block=True)