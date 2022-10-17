import numpy as np

"""
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
"""


"""
Computation on NumPy arrays can be very fast, or it can be very slow. 
The key to making it fast is to use vectorized operations, generally 
implemented through NumPy's universal functions (ufuncs). 
"""

"""
Python uses VECTORIZED OPERATIONS which consist on apply an operation
to each element of the array
"""

x = np.arange(9).reshape((3, 3))
2 ** x

x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division (Truncate this number)
print("x ** 2 =", x ** 2)
print("x % 2 =", x % 2)

# ABSOLUTE VALUE

x = np.array([-2, -1, 0, 1, 2])
abs(x)

np.abs(x)

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)

"""
When is a complex this will return the magnitude
"""

# TRIGONOMETRIC FUNCTIONS 

theta = np.linspace(0, np.pi, 3)

print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

print("arcsin(theta) = ", np.arcsin(x))
print("arccos(theta) = ", np.arccos(x))
print("arctan(theta) = ", np.arctan(x))

# EXPONENTS AND LOGARITHMS

x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))


x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))


# SPECIALIZED FUNCTIONS 

from scipy import special

x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))

# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)  =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


# SPECIFYING OUTPUT WHEN WE USE FUNCTIONS

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

"""
Or in a sub-array like this
"""
y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

# AGGREGATES 

x = np.arange(1, 6)
np.add.reduce(x)        # REGRESA LA SUMA DE TODOS

np.multiply.reduce(x)   # REGRESA EL PRODUCTO DE TODOS

np.add.accumulate(x)
np.multiply.accumulate(x)

"""
With accumulate we stored into an array
"""