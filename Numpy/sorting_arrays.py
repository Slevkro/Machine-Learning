import numpy as np 

# SORTING

x = np.array([2, 1, 4, 3, 5])
np.sort(x)

"""Can aslo use"""

x.sort()
print(x)

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
x[i]

"""
argsort -> Returns the indices of sorted elements 
            in order to use fancy indexing
"""

# ALONG ROWS AND COLUMNS

rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)

# sort each column of X
np.sort(X, axis=0)

# sort each row of X
np.sort(X, axis=1)

"""
With axis as parameters we sort rows or columns
0 -> Column
1 -> Row
"""

# PARTIAL SORTS 

x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)

"""
partition -> takes the k + 1 smallest elements and set them on 
            the left in tis case 3 elements

"""

# K-NEAREST NEIGHBORS

X = rand.rand(10, 2)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], s=100);

dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)

# xd

# for each pair of points, compute differences in their coordinates
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape

"""
(10, 10, 2)

Each with the difference to each point
"""
# square the coordinate differences
sq_differences = differences ** 2
sq_differences.shape

# sum the coordinate differences to get the squared distance
dist_sq = sq_differences.sum(-1)
dist_sq.shape

dist_sq.diagonal()

nearest = np.argsort(dist_sq, axis=1)
print(nearest)

K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

"""
Cambia a la segunda columna que es la que nos importa
"""

plt.scatter(X[:, 0], X[:, 1], s=100)

# draw lines from each point to its two nearest neighbors
K = 2

for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')
plt.show(block = True)
plt.show(block=True)

# CREATE STRUCTURED ARRAYS

x = np.zeros(4, dtype = int)

data = np.zeros(4, dtype={
    'names':('name', 'age', 'weight'),
    'formats':('U10', 'i4', 'f8')
})

data = np.zeros(4, dtype={
    'names':('name', 'age', 'weight'),
    'formats':('U10', 'i4', 'f8')
})

"""
U10 -> Unicode string of max length 10
i4 -> 4-byte integer
f8 -> 8-byte float
"""

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data['name'] = name 
data['age'] = age 
data['weight'] = weight

print(data)

# Get first row of data
data[0]

# Get the name from the last row
data[-1]['name']

# Get names where age is under 30
data[data['age'] < 30]['name']

# CREATING STRUCTURES ARRAYS

np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})

np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})


# RECORD ARRAYS

data_rec = data.view(np.recarray)
data_rec.age

"""
Verla como una matriz de registros
"""
