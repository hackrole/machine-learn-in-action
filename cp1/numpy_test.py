from numpy import *

# generate a 4x4 array
print(random.rand(4, 4))

# generate a 4x4 matrix
randMat = mat(random.rand(4, 4))
print(randMat)

# get the matrix's oppodsites matrix and mat them.
# the should generate a cell martrix if not the math errors.
print(randMat.I)
print(randMat * randMat.I)

# generate a cell matric
print(eye(4))
