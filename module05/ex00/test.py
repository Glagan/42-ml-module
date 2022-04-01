from matrix import Matrix, Vector

print('## Matrix')
print('# Constructors')
print(repr(Matrix([[1.0]])))
print(repr(Matrix([[1.0], [1.0]])))
print(repr(Matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])))
print(repr(Matrix((1, 1))))
print(repr(Matrix((6, 6))))
print(repr(Matrix([[1.0]], (1, 1))))
print(repr(Matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], (2, 3))))

print('\n# Transpose')
m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
print('m1')
print(repr(m1))
print('m1.T()')
print(repr(m1.T()))
print('m1.T().T()')
print(repr(m1.T().T()))


m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
print('\n## Scalar')
print('m1 {}'.format(m1))
print('# Addition')
print(m1 + 1)
print(1 + m1)
print(m1 + 2.5)
print(2.5 + m1)
print('# Substraction')
print(m1 - 1)
print(1 - m1)
print(m1 - 2.5)
print(2.5 - m1)
print('# Multiplication')
print(m1 * 2)
print(2 * m1)
print(m1 * 2.5)
print(2.5 * m1)
print('# Division')
print(m1 / 2)
print(m1 / 2.5)

print('\n# Errors')
try:
    m0 = Matrix()
except Exception as err:
    print(err)
try:
    m0 = Matrix('matrix')
except Exception as err:
    print(err)
try:
    m0 = Matrix((0, 1))
except Exception as err:
    print(err)
try:
    m0 = Matrix((1, 0))
except Exception as err:
    print(err)
try:
    m0 = Matrix((12))
except Exception as err:
    print(err)
try:
    m0 = Matrix(('height'))
except Exception as err:
    print(err)
try:
    m0 = Matrix(('height', 'width'))
except Exception as err:
    print(err)
try:
    m0 = Matrix([])
except Exception as err:
    print(err)
try:
    m0 = Matrix([12])
except Exception as err:
    print(err)
try:
    m0 = Matrix([[12]])
except Exception as err:
    print(err)
try:
    m0 = Matrix([[12.0, 42]])
except Exception as err:
    print(err)
try:
    m0 = Matrix([[12.0, 42.0], [12]])
except Exception as err:
    print(err)
try:
    m0 = Matrix([[12.0, 42.0], [12.0, 42]])
except Exception as err:
    print(err)
try:
    m0 = Matrix(['quarante-deux'])
except Exception as err:
    print(err)
try:
    m0 = Matrix([['quarante-deux']])
except Exception as err:
    print(err)
try:
    m0 = Matrix([[12.0, 'quarante-deux', 6.0]])
except Exception as err:
    print(err)
try:
    print(2 / m1)
except Exception as err:
    print(err)
try:
    print(2.5 / m1)
except Exception as err:
    print(err)
try:
    print(m1 / 0)
except Exception as err:
    print(err)
try:
    print(m1 / 0.0)
except Exception as err:
    print(err)

v1 = Vector([[1.0], [2.0], [4.0], [8.0]])
print('\n## Vector')
print('v1 {}'.format(m1))
print('v2 {}'.format(v1))
print('# Multiplication')
print(m1 * v1)

v2 = Vector([[1.0], [2.0]])
print('\n# Errors')
print('v2 {}'.format(v2))
try:
    print(v1 * m1)
except Exception as err:
    print(err)
try:
    print(m1 * v2)
except Exception as err:
    print(err)
try:
    print(v2 * m1)
except Exception as err:
    print(err)

m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
m3 = Matrix([[2.0, 4.0, 6.0, 8.0], [-2.0, -4.0, -6.0, -8.0]])
m4 = Matrix([[-1.0, 1.0]])

print('\n## Matrix')
print('m1 {}'.format(m1))
print('m2 {}'.format(m2))
print('m3 {}'.format(m3))
print('# Addition')
print(m1 + m3)
print(m3 + m1)
print('# Substraction')
print(m1 - m3)
print(m3 - m1)
print('# Multiplication')
print(m1 * m2)
print(m2 * m1)

print('\n# Errors')
print('4 {}'.format(m4))
try:
    print(m1 + m2)
except Exception as err:
    print(err)
try:
    print(m2 + m1)
except Exception as err:
    print(err)
try:
    print(m1 - m2)
except Exception as err:
    print(err)
try:
    print(m2 - m1)
except Exception as err:
    print(err)
try:
    print(m1 * m4)
except Exception as err:
    print(err)

print('\n## Vector')
print('# Constructors')
print(repr(Vector([[1.0], [2.0], [4.0], [8.0]])))
print(repr(Vector([[1.0]])))
print(repr(Vector([[1.0], [1.0]])))
print(repr(Vector([[-3.0], [-2.0], [-1.0], [0.0]])))
print(repr(Vector([1.0])))
print(repr(Vector([1.0, 2.0, 3.0, 4.0])))

print('\n# Errors')
try:
    v0 = Vector([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
except Exception as err:
    print(err)
try:
    v0 = Vector([['42'], [1.0]])
except Exception as err:
    print(err)
try:
    v0 = Vector([[1.0, 2.0], [1.0]])
except Exception as err:
    print(err)
try:
    v0 = Vector([[1.0], [1.0, 2.0]])
except Exception as err:
    print(err)
try:
    v0 = Vector([1.0, '42', 42.0])
except Exception as err:
    print(err)

print('\n# Transpose')
v1 = Vector([1.0, 2.0, 3.0, 4.0])
print('v1')
print(repr(v1))
print('v1.T()')
print(repr(v1.T()))
print('v1.T().T()')
print(repr(v1.T().T()))

print('\n# dot')
v = Vector([[0.0], [1.0], [2.0], [3.0]])
v2 = Vector([0.0, 1.0, 2.0, 3.0])
print("(0*0) + (1*1) + (2*2) + (3*3) = 14")
print("v.dot(v2.T()) = {}".format(v.dot(v2.T())))

v1 = Vector([0.0, 1.0, 2.0, 3.0])
print('\n## Scalar')
print('v1 {}'.format(v1))
print('# Addition')
print(v1 + 1)
print(1 + v1)
print(v1 + 2.5)
print(2.5 + v1)
print('# Substraction')
print(v1 - 1)
print(1 - v1)
print(v1 - 2.5)
print(2.5 - v1)
print('# Multiplication')
print(v1 * 2)
print(2 * v1)
print(v1 * 2.5)
print(2.5 * v1)
print('# Division')
print(v1 / 2)
print(v1 / 2.5)
