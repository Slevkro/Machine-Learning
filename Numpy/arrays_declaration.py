import numpy as np 

"""
Instalacion: 
sudo apt install python3-numpy
pip3 install --upgrade numpy
"""

# integer array:
print(np.array([1, 4, 2, 5, 3]))

# numpy will up-cast if possible
print(np.array([3.14, 4, 2, 3]))

# dtype fpr set a data type
print(np.array([1, 2, 3, 4], dtype='float32'))

# nested lists result in multi-dimensional arrays
print(np.array([range(i, i + 3) for i in [2, 4, 6]]))

"""
method array to create numpy arrays with a list
"""

print('--------------------------------')
print('Creando arrays desde cero')

# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)

# Create a 3x5 floating-point array filled with ones
# 3-> vertical, 5->horizontal
np.ones((3, 5), dtype=float)

# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)

# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
np.arange(0, 20, 2)

# Create an array of five values evenly spaced between 0 and 1
# splits on five spaces between 0 to 1
np.linspace(0, 1, 5)

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))

# Create a 3x3 identity matrix
np.eye(3)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)

"""
zeros, ones -> to create an array with zeros or ones
full -> to create an array full of the specified number as second parameter
arrange -> to create an array that follows a range
linespace -> to create an array split on equals parts
random.random, normal, randint -> to create an array with random numbers 
eye -> to create an identity matrix
empty -> to create an empty matrix filled with garbage
"""

#===============================================================================================
#===============================================================================================
#===============================================================================================
#===============================================================================================
#===============================================================================================

# DESCRIBIND ARRAYS

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)

"""
ndim -> number of dimensions (3)
shape -> size of each dimension (3x4x5)
size -> number of elements 3x4x5 = 60
dtype -> datatype
"""

# INDEXING ARRAYS

x1[4]   # print the five item
x1[-1]  # last item
x1[-2]  # penultimate element

# In multidimension arrays 

"""
x2
array([[3, 5, 2, 4],
       [7, 6, 8, 8],
       [1, 6, 7, 7]])
"""

x2[0, 0]  # (0, 0) prints 3
x2[0, 0] = 12  # Modify 3 to 12

x2[2, 0]  # (2, 0) prints 1


# First vertical then horizontal

x2[2, -1]  # (2, -1) prints 7

x1[0] = 3.14159  # this will be truncated!

# SLICING ARRAYS

# x[start:stop:step]

x = np.arange(10)

x[:5]  # first five elements

x[5:]  # elements after index 5

x[4:7]  # middle sub-array, from 5 to 6 index

x[::2]  # every two

x[1::2]  # from 1 to 10 every 2 (without 0)

x[::-1]  # all elements, reversed

"""
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
"""

# Same but row-columns

x2[:2, :3]  # two rows, three columns
# columns 0, 1 and rows 0, 1, 2

x2[:3, ::2]  # all rows, every two columns

print(x2[:, 0])  # first column of x2

print(x2[0, :])  # first row of x2

x2_sub_copy = x2[:2, :2].copy()  # a copy of first two rows and two columns

"""
We follow [start:stop:step]

in two dimmensions [row -> [start:stop:step], column ->[start:stop:step]]

If we take a sub array as a copy and then modify it also we will be
modifying the original but with copy() method we can modify without 
change the original
"""

x2_sub_copy = x2[:2, :2].copy()

# RESHAPING ARRAYS

grid = np.arange(1, 10)
print(grid)     # 1x9 
grid = grid.reshape((3, 3))
print(grid)     # 3x3

# Must fit the size of both arrays in other words 1x9 = 3x3

x = np.array([1, 2, 3])

# row vector via reshape
x.reshape((1, 3))

# row vector via newaxis
x[np.newaxis, :]

# column vector via reshape
x.reshape((3, 1))

# column vector via newaxis
x[:, np.newaxis]

"""
reshape -> to change the shape of an existing array but must fit in size
"""

# ARRAY CONCATENATION

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

#Two dimension

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

np.concatenate([grid, grid])    # It concatenates in vertical

np.concatenate([grid, grid], axis=1)    # It concatenates in horizontal

"""
For working with arrays of mixed dimensions, it can be clearer to use 
the np.vstack (vertical stack) and np.hstack (horizontal stack) functions

Similary, np.dstack will stack arrays along the third axis.
"""

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
# vertically stack the arrays
np.vstack([x, grid])
np.vstack([grid, x])

y = np.array([[99],
              [99]])
# horizontally stack the arrays
np.hstack([grid, y])
np.hstack([grid, x])

"""
v/hstack([up-element, bottom-element])

The opposite of concatenation is splitting, which is implemented by 
the functions np.split, np.hsplit, and np.vsplit.
"""

# FANCY INDEXING


import numpy as np
rand = np.random.RandomState(42)

x = rand.randint(100, size=10)
print(x)

ind = np.array([[3, 7],
                [4, 5]])

print(x[ind])

"""
array([[71, 86],
       [60, 20]])

Takes the indexes in a certain shape
"""

X = np.arange(12).reshape((3, 4))
X

row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

"""
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

array([ 2,  5, 11])
"""

X[2, [2, 0, 1]]

"""
Row 2 and columns from 2 to 0 
"""

# RANDOM POINTS

mean = [0, 0]

cov = [[1, 2], 
       [2, 5]]

X = rand.multivariate_normal(mean, cov, 100)

X.shape

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])

plt.show(block = True)

indices = np.random.choice(X.shape[0], 20, replace = False)

print(indices)

sample = X[indices]

plt.scatter(X[:, 0], X[:, 1], alpha = 0.3)
plt.scatter(sample[:, 0], sample[:, 1], facecolor = 'none', s = 200)
plt.show(block = True)

"""
Here we use to select a random sample 
"""

x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)

"""
Also works to modify values
"""


i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x             # DOES NOT MODIFY

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)      # MODIFYES

# BINNING DATA

np.random.seed(42)
x = np.random.randn(100)

# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)

# add 1 to each of these bins
np.add.at(counts, i, 1)


# plot the results
plt.plot(bins, counts, linestyle='steps');




