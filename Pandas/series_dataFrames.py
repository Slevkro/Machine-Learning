import numpy as np
import pandas as pd


# PANDAS SERIES

"""
Are one-dimensional arrays of index data
pd.Series(data, index=index)
index by default is numeric but can be a list f anything
"""

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)

print(data.values)
print(data.index)

"""Access and defined indexes"""

data[1]
data[1:3]

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print(data)

print(data['b'])

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])


"""Series from dictionaries"""

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}

population = pd.Series(population_dict)

print(population)

print(population['California':'Illinois'])

"""
Sort it
California    38332521
Florida       19552860
Illinois      12882135
New York      19651127
Texas         26448193
dtype: int64
"""

# PANDAS DATA FRAMES

"""
Are two-dimensional arrays of index data 

With both flexible row indices and flexible column names.
"""

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area

states = pd.DataFrame({'population': population, 'area':area})
print(states)

print(states.index)
print(states.columns)

print(states['area'])

"""Construct Data Frames Objects"""

# From a series

pd.DataFrame(population, columns=['population'])

# From a lis of dicts

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

# From a dictionary of series objects

pd.DataFrame({'population': population,
              'area': area})

# From a numpy array 

pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])

# Pandas index array

"""
Index as an immutable array and similar to NumPy arrays
"""

ind = pd.Index([2, 3, 5, 7, 11])
ind

print(ind.size, ind.shape, ind.ndim, ind.dtype)

ind[1]
ind[::2]

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])

indA & indB  # intersection

indA | indB  # union

indA ^ indB  # symmetric difference