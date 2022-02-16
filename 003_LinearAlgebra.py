# Vectors
# create a vector
from numpy.linalg import eig
from scipy.linalg import svd
from scipy.linalg import lu
from numpy.linalg import inv
from numpy import array
v = array([1, 2, 3])
print(v)

# multiply vectors
print('Multiply vectors')
a = array([1, 2, 3])
print(a)
b = array([1, 2, 3])
print(b)
c = a * b
print(c)

# create matrix
print("Matrix creation")
A = array([[1, 2, 3], [4, 5, 6]])
print(A)

# add matrices
print("Matrix addition")
A = array([[1, 2, 3], [4, 5, 6]])
print(A)
B = array([[1, 2, 3], [4, 5, 6]])
print(B)
C = A + B
print(C)

# matrix dot product
print("Matrix multiplication - dot product")
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
B = array([[1, 2], [3, 4]])
print(B)
C = A.dot(B)
print(C)

# transpose matrix
print("Transpose matrix")
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
C = A.T
print(C)

# invert matrix
print('Invert matrix')
# define matrix
A = array([[1.0, 2.0], [3.0, 4.0]])
print(A)
# invert matrix
B = inv(A)
print(B)

# LU decomposition
print('Matrix decomposition')
# define a square matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# LU decomposition
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
B = P.dot(L).dot(U)
print(B)

# Singular-value decomposition
print('Singular-value decomposition')
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# SVD
U, s, V = svd(A)
print(U)
print(s)
print(V)

# Eigen decomposition
print('Eigen decomposition')
# # define matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# calculate eigendecomposition
values, vectors = eig(A)
print(values)
print(vectors)
