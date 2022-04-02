from matrix import Matrix, Vector

print('### Matrix')

print('## Constructors')
print(repr(Matrix([[1.0]])))
print(repr(Matrix([[1.0], [1.0]])))
print(repr(Matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])))
print(repr(Matrix((1, 1))))
print(repr(Matrix((6, 6))))
print(repr(Matrix([[1.0]], (1, 1))))
print(repr(Matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], (2, 3))))

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

print('\n## Transpose')
m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
m1T = Matrix([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
print('m1')
print(repr(m1))
print('m1.T()')
print(repr(m1.T()))
assert (m1.T()).data == m1T.data, 'Transpose failed'
print('m1.T().T()')
print(repr(m1.T().T()))
assert (m1.T().T()).data == m1.data, 'Transpose failed'


m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
print('\n## Scalar')
print('m1 {}'.format(m1))

print('# Addition')
m1plus1 = Matrix([[1.0, 2.0, 3.0, 4.0],
                 [1.0, 3.0, 5.0, 7.0]])
m1plus2 = Matrix([[2.5, 3.5, 4.5, 5.5],
                 [2.5, 4.5, 6.5, 8.5]])
print('m1 + 1', m1 + 1)
assert (m1 + 1).data == m1plus1.data, '[left] Addition failed'
print('1 + m1', 1 + m1)
assert (1 + m1).data == m1plus1.data, '[right] Addition failed'
print('m1 + 2.5', m1 + 2.5)
assert (m1 + 2.5).data == m1plus2.data, '[left] Addition failed'
print('2.5 + m1', 2.5 + m1)
assert (2.5 + m1).data == m1plus2.data, '[right] Addition failed'

print('# Substraction')
m1minus1 = Matrix([[-1.0, 0.0, 1.0, 2.0], [-1.0, 1.0, 3.0, 5.0]])
m1minus2 = Matrix([[-2.5, -1.5, -0.5, 0.5], [-2.5, -0.5, 1.5, 3.5]])
print('m1 - 1', m1 - 1)
assert (m1 - 1).data == m1minus1.data, '[left] Substraction failed'
print('1 - m1', 1 - m1)
assert (1 - m1).data == m1minus1.data, '[right] Substraction failed'
print('m1 - 2.5', m1 - 2.5)
assert (m1 - 2.5).data == m1minus2.data, '[left] Substraction failed'
print('2.5 - m1', 2.5 - m1)
assert (2.5 - m1).data == m1minus2.data, '[right] Substraction failed'

print('# Multiplication')
m1times2 = Matrix([[0.0, 2.0, 4.0, 6.0], [0.0, 4.0, 8.0, 12.0]])
m1times2half = Matrix([[0.0, 2.5, 5.0, 7.5], [0.0, 5.0, 10.0, 15.0]])
print('m1 * 2', m1 * 2)
assert (m1 * 2).data == m1times2.data, '[left] Multiplication failed'
print('2 * m1', 2 * m1)
assert (2 * m1).data == m1times2.data, '[right] Multiplication failed'
print('m1 * 2.5', m1 * 2.5)
assert (m1 * 2.5).data == m1times2half.data, '[left] Multiplication failed'
print('2.5 * m1', 2.5 * m1)
assert (2.5 * m1).data == m1times2half.data, '[right] Multiplication failed'

print('# Division')
m1div2 = Matrix([[0.0, 0.5, 1.0, 1.5], [0.0, 1.0, 2.0, 3.0]])
m1div2half = Matrix([[0.0, 0.4, 0.8, 1.2], [0.0, 0.8, 1.6, 2.4]])
print('m1 / 2', m1 / 2)
assert (m1 / 2).data == m1div2.data, '[left] Division failed'
print('m1 / 2.5', m1 / 2.5)
assert (m1 / 2.5).data == m1div2half.data, '[left] Division failed'

print('\n# Errors')
try:
    print(m1 + '2')
except Exception as err:
    print(err)
try:
    print('2' + m1)
except Exception as err:
    print(err)
try:
    print(m1 - '2')
except Exception as err:
    print(err)
try:
    print('2' - m1)
except Exception as err:
    print(err)
try:
    print(m1 * '2')
except Exception as err:
    print(err)
try:
    print('2' * m1)
except Exception as err:
    print(err)
try:
    print(m1 / '2')
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

m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
m3 = Matrix([[2.0, 4.0, 6.0, 8.0], [-2.0, -4.0, -6.0, -8.0]])
print('\n## Matrix')
print('m1 {}'.format(m1))
print('m2 {}'.format(m2))
print('m3 {}'.format(m3))

print('# Addition')
m1plusm3 = Matrix([[2.0, 5.0, 8.0, 11.0], [-2.0, -2.0, -2.0, -2.0]])
print('m1 + m3', m1 + m3)
assert (m1 + m3).data == m1plusm3.data, '[left] Addition failed'
print('m3 + m1', m3 + m1)
assert (m3 + m1).data == m1plusm3.data, '[right] Addition failed'

print('# Substraction')
m1minusm3 = Matrix([[-2.0, -3.0, -4.0, -5.0], [2.0, 6.0, 10.0, 14.0]])
m3minusm1 = Matrix([[2.0, 3.0, 4.0, 5.0], [-2.0, -6.0, -10.0, -14.0]])
print('m1 - m3', m1 - m3)
assert (m1 - m3).data == m1minusm3.data, '[left] Substraction failed'
print('m3 - m1', m3 - m1)
assert (m3 - m1).data == m3minusm1.data, '[right] Substraction failed'

print('# Multiplication')
m1timesm2 = Matrix([[28.0, 34.0], [56.0, 68.0]])
m2timesm1 = Matrix([[0.0, 2.0, 4.0, 6.0], [0.0, 8.0, 16.0, 24.0],
                    [0.0, 14.0, 28.0, 42.0], [0.0, 20.0, 40.0, 60.0]])
print('m1 * m2', m1 * m2)
assert (m1 * m2).data == m1timesm2.data, '[left] Multiplication failed'
print('m2 * m1', m2 * m1)
assert (m2 * m1).data == m2timesm1.data, '[right] Multiplication failed'

m4 = Matrix([[-1.0, 1.0]])
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

m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
v1 = Vector([[1.0], [2.0], [4.0], [8.0]])
print('\n## Vector')
print('v1 {}'.format(m1))
print('v2 {}'.format(v1))

print('# Multiplication')
m1timesv1 = Matrix([[34.0], [68.0]])
print('m1 * v1', m1 * v1)
assert (m1 * v1).data == m1timesv1.data, 'Vector multiplication failed'

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

print('\n\n### Vector')

print('## Constructors')
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

v1 = Vector([1.0, 2.0, 3.0, 4.0])
v1T = Matrix([[1.0], [2.0], [3.0], [4.0]])
print('\n## Transpose')
print('v1')
print(repr(v1))
print('v1.T()')
print(repr(v1.T()))
assert (v1.T()).data == v1T.data, 'Transpose failed'
print('v1.T().T()')
print(repr(v1.T().T()))
assert (v1.T().T()).data == v1.data, 'Transpose failed'

print('\n## dot')
v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
v2 = Vector([0.0, 1.0, 2.0, 3.0])
print("(0*0) + (1*1) + (2*2) + (3*3) = 14")
print("v1.dot(v2.T()) = {}".format(v1.dot(v2.T())))

v3 = Vector([1.0])
print('\n# Errors')
print('v3 {}'.format(v3))
try:
    v1.dot('42')
except BaseException as err:
    print(err)
try:
    v1.dot(42)
except BaseException as err:
    print(err)
try:
    v1.dot(v3)
except BaseException as err:
    print(err)

v1 = Vector([0.0, 1.0, 2.0, 3.0])
print('\n## Scalar')
print('v1 {}'.format(v1))

print('# Addition')
v1plus1 = Vector([1.0, 2.0, 3.0, 4.0])
v1plus2half = Vector([2.5, 3.5, 4.5, 5.5])
print(v1 + 1)
assert isinstance(v1 + 1, Vector), 'Addition result is not a Vector'
assert (v1 + 1).data == v1plus1.data, '[left] Addition failed'
print(1 + v1)
assert isinstance(1 + v1, Vector), 'Addition result is not a Vector'
assert (1 + v1).data == v1plus1.data, '[right] Addition failed'
print(v1 + 2.5)
assert isinstance(v1 + 2.5, Vector), 'Addition result is not a Vector'
assert (v1 + 2.5).data == v1plus2half.data, '[left] Addition failed'
print(2.5 + v1)
assert isinstance(2.5 + v1, Vector), 'Addition result is not a Vector'
assert (2.5 + v1).data == v1plus2half.data, '[right] Addition failed'

print('# Substraction')
v1minus1 = Vector([-1.0, 0.0, 1.0, 2.0])
v1minus2half = Vector([-2.5, -1.5, -0.5, 0.5])
print(v1 - 1)
assert isinstance(v1 - 1, Vector), 'Substraction result is not a Vector'
assert (v1 - 1).data == v1minus1.data, '[left] Substraction failed'
print(1 - v1)
assert isinstance(1 - v1, Vector), 'Substraction result is not a Vector'
assert (1 - v1).data == v1minus1.data, '[right] Substraction failed'
print(v1 - 2.5)
assert isinstance(v1 - 2.5, Vector), 'Substraction result is not a Vector'
assert (v1 - 2.5).data == v1minus2half.data, '[left] Substraction failed'
print(2.5 - v1)
assert isinstance(2.5 - v1, Vector), 'Substraction result is not a Vector'
assert (2.5 - v1).data == v1minus2half.data, '[right] Substraction failed'

print('# Multiplication')
v1times2 = Vector([0.0, 2.0, 4.0, 6.0])
v1times2half = Vector([0.0, 2.5, 5.0, 7.5])
print(v1 * 2)
assert isinstance(v1 * 2, Vector), 'Multiplication result is not a Vector'
assert (v1 * 2).data == v1times2.data, '[left] Multiplication failed'
print(2 * v1)
assert isinstance(2 * v1, Vector), 'Multiplication result is not a Vector'
assert (2 * v1).data == v1times2.data, '[right] Multiplication failed'
print(v1 * 2.5)
assert isinstance(v1 * 2.5, Vector), 'Multiplication result is not a Vector'
assert (v1 * 2.5).data == v1times2half.data, '[left] Multiplication failed'
print(2.5 * v1)
assert isinstance(2.5 * v1, Vector), 'Multiplication result is not a Vector'
assert (2.5 * v1).data == v1times2half.data, '[right] Multiplication failed'

print('# Division')
v1div2 = Vector([0.0, 0.5, 1.0, 1.5])
v1div2half = Vector([0.0, 0.4, 0.8, 1.2])
print(v1 / 2)
assert isinstance(v1 / 2, Vector), 'Division result is not a Vector'
assert (v1 / 2).data == v1div2.data, '[left] Division failed'
print(v1 / 2.5)
assert isinstance(v1 / 2.5, Vector), 'Division result is not a Vector'
assert (v1 / 2.5).data == v1div2half.data, '[left] Division failed'

v1 = Vector([0.0, 1.0, 2.0, 3.0])
print('\n# Errors')
try:
    print(v1 + '2')
except Exception as err:
    print(err)
try:
    print('2' + v1)
except Exception as err:
    print(err)
try:
    print(v1 - '2')
except Exception as err:
    print(err)
try:
    print('2' - v1)
except Exception as err:
    print(err)
try:
    print(v1 * '2')
except Exception as err:
    print(err)
try:
    print('2' * v1)
except Exception as err:
    print(err)
try:
    print(v1 / '2')
except Exception as err:
    print(err)
try:
    print(2 / v1)
except Exception as err:
    print(err)
try:
    print(2.5 / v1)
except Exception as err:
    print(err)
try:
    print(v1 / 0)
except Exception as err:
    print(err)
try:
    print(v1 / 0.0)
except Exception as err:
    print(err)

v1 = Vector([0.0, 1.0, 2.0, 3.0])
v2 = Vector([0.0, 2.0, 4.0, 6.0])
v3 = Vector([[0.0], [2.0], [4.0], [6.0]])
print('\n## Vector')
print('v1 {}'.format(v1))
print('v2 {}'.format(v2))

print('# Addition')
v1plusv2 = Vector([0.0, 3.0, 6.0, 9.0])
print('v1 + v2', v1 + v2)
assert isinstance(v1 + v2, Vector), 'Addition result is not a Vector'
assert (v1 + v2).data == v1plusv2.data, 'Addition failed'
print('v2 + v1', v2 + v1)
assert isinstance(v2 + v1, Vector), 'Addition result is not a Vector'
assert (v2 + v1).data == v1plusv2.data, 'Addition failed'

print('# Substraction')
v1minusv2 = Vector([0.0, -1.0, -2.0, -3.0])
v2minusv1 = Vector([0.0, 1.0, 2.0, 3.0])
print('v1 - v2', v1 - v2)
assert isinstance(v1 + v2, Vector), 'Substraction result is not a Vector'
assert (v1 + v2).data == v1plusv2.data, 'Substraction failed'
print('v2 - v1', v2 - v1)
assert isinstance(v1 + v2, Vector), 'Substraction result is not a Vector'
assert (v1 + v2).data == v1plusv2.data, 'Substraction failed'

print('# Multiplication')
v1timesv3 = Vector([[28.0]])
v3timesv1 = Vector([0.0, 0.0, 0.0, 0.0])
print('v1 * v3', v1 * v3)
assert isinstance(v1 * v3, Vector), 'Multiplication result is not a Vector'
assert (v1 * v3).data == v1timesv3.data, 'Multiplication failed'
print('v3 * v1', v3 * v1)
assert isinstance(v3 * v1, Vector), 'Multiplication result is not a Vector'
assert (v3 * v1).data == v3timesv1.data, 'Multiplication failed'

v1 = Vector([0.0, 1.0, 2.0, 3.0])
v4 = Vector([-1.0, 1.0])
print('\n# Errors')
print('v4 {}'.format(v4))
try:
    print(v1 + v4)
except Exception as err:
    print(err)
try:
    print(v4 + v1)
except Exception as err:
    print(err)
try:
    print(v1 - v4)
except Exception as err:
    print(err)
try:
    print(v4 - v1)
except Exception as err:
    print(err)
try:
    print(v1 * v4)
except Exception as err:
    print(err)
try:
    print(v4 * v1)
except Exception as err:
    print(err)
