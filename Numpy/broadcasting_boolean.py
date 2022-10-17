import numpy as np 

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])


a + 5

"""
We can think of this as an operation that stretches or 
duplicates the value 5 into the array [5, 5, 5], and 
adds the results.
"""

M = np.ones((3, 3))

M + a

# broadcast across the second dimension in order to match the shape of M

# RULE 1 AND 2

M = np.ones((2, 3))
a = np.arange(3)

M + a

"""
M.shape = (2, 3)
a.shape = (3,)

so we pad it on the left with ones
M.shape -> (2, 3)
a.shape -> (1, 3)

Then by rule 2
M.shape -> (2, 3)
a.shape -> (2, 3)

"""

# RULE 3

M = np.ones((3, 2))
a = np.arange(3)

# M + a 

"""
M = np.ones((3, 2))
a = np.arange(3)

Rule 1 tells us that we must pad the shape of 'a' with ones

M.shape -> (3, 2)
a.shape -> (1, 3)

By rule 2, the first dimension of 'a' is stretched to match that of M

M.shape -> (3, 2)
a.shape -> (3, 3)

Now we hit rule 3–the final shapes do not match, so these two arrays are incompatible

"""

# PRACTICAL EXAMPLE (CENTERING AN ARRAY)

X = np.random.random((10, 3))

Xmean = X.mean(0)   # La media de las 3 columnas
Xmean

X_centered = X - Xmean


# BOOLEAN LOGIC

import pandas as pd
import matplotlib.pyplot as plt

# use pandas to extract rainfall inches as a NumPy array
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0  # 1/10mm -> inches
print(inches.shape)
# print(inches)

"""
The array contains 365 values, giving daily rainfall 
in inches from January 1 to December 31, 2014.
"""

plt.hist(inches, 40)
plt.show(block=True)

x = np.array([1, 2, 3, 4, 5])
print(x < 3)
print((2 * x) == (x ** 2))

"""
We can use 
< 
> 
<= 
>= 
!= 
==
"""

# COUNTING ARROUND BOOLEAN ARRAYS

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))

np.count_nonzero(x < 6)
np.sum(x < 6)

# how many values less than 6 in each row?
np.sum(x < 6, axis=1)

"""
count_nonzero(...) counts the number of true
Another way is with sum
"""

# are there any values greater than 8?
np.any(x > 8)       # True

# are all values less than 10?
np.all(x < 10)      # False

# are all values in each row less than 8?
np.all(x < 8, axis=1)

"""
any -> if one is True

all -> if all are true
"""


# BOOLEAN OPERATORS

print(np.sum((inches > 0.5) & (inches < 1)))

print("Number days without rain:      ", np.sum(inches == 0))
print("Number days with rain:         ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches  :", np.sum((inches > 0) &
                                                (inches < 0.2)))



# BOOLEAN ARRAYS AS MASKS

x_mask = x < 5

"""
This returns a boolean array aka mask
"""

# construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2014 (inches):   ",
      np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):  ",
      np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ",
      np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):",
      np.median(inches[rainy & ~summer]))
