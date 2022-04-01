class Matrix:
    def __init__(self, initializer, shape=None) -> None:
        self.data = []
        if shape == None:
            if isinstance(initializer, list):
                self.shape = None
                if not initializer:
                    raise TypeError("A Matrix can't be smaller than 1x1.")
                for col in initializer:
                    if not isinstance(col, list):
                        raise TypeError(
                            'Each columns in a row must be a list.')
                    if not self.shape:
                        if not col:
                            raise TypeError(
                                "A Matrix can't be smaller than 1x1.")
                        self.shape = (len(initializer), len(col))
                    elif self.shape[1] != len(col):
                        raise TypeError(
                            'Each columns must have the same length.')
                    self.data.append([])
                    for value in col:
                        if type(value) is not float:
                            raise TypeError(
                                'Each values in the matrix must be a float.')
                        self.data[-1].append(value)
            elif isinstance(initializer, tuple):
                if len(initializer) != 2 or not type(initializer[0]) is int or not type(initializer[1]) is int:
                    raise TypeError(
                        'Expected a (min, max) tuple as a shape got {}.'.format(initializer))
                if initializer[0] < 1 or initializer[1] < 1:
                    raise TypeError("A Matrix can't be smaller than 1x1.")
                self.data = [[0.0] * initializer[1]] * initializer[0]
                self.shape = (initializer[0], initializer[1])
            else:
                raise TypeError(
                    'Matrix initializer must be a list of list or a shape.')
        else:
            if not isinstance(initializer, list):
                raise TypeError(
                    'Matrix initializer must be a list of list of the given shape')
            if not isinstance(shape, tuple) or len(shape) != 2 or not type(shape[0]) is int or not type(shape[1]) is int:
                raise TypeError(
                    'Expected a (min, max) tuple as a shape got {}.'.format(initializer))
            if shape[0] < 1 or shape[1] < 1:
                raise TypeError("A Matrix can't be smaller than 1x1.")
            self.shape = (shape[0], shape[1])
            self.data = []
            if len(initializer) != shape[0]:
                raise TypeError('Invalid list for given shape.')
            for col in initializer:
                if not isinstance(col, list):
                    raise TypeError('Each columns in a row must be a list.')
                if self.shape[1] != len(col):
                    raise TypeError('Invalid column length for given shape.')
                self.data.append([])
                for value in col:
                    if type(value) is not float:
                        raise TypeError(
                            'Each values in the matrix must be a float.')
                    self.data[-1].append(value)

    def T(self):
        # Reverse shape
        m, n = self.shape
        copy = [[[] for _ in range(m)] for _ in range(n)]
        # Loop in current shape order and append in new shape order
        x, y = 0, 0
        for row in self.data:
            for col in row:
                copy[x][y] = col
                x = (x + 1) % n
                if x == 0:
                    y = y + 1
        if isinstance(self, Vector):
            # Converted to row vector
            if n == 1:
                return Vector(copy[0])
            return Vector(copy)
        return Matrix(copy)

    def __add__(self, b):
        if type(b) is int or type(b) is float:
            result = []
            for row in self.data:
                result.append(list(col + b for col in row))
            return Matrix(result, self.shape)
        elif isinstance(b, Matrix):
            if self.shape != b.shape:
                raise TypeError('You can only add Matrices of the same size.')
            result = []
            for row in range(self.shape[0]):
                result.append(
                    list(self.data[row][col] + b.data[row][col] for col in range(self.shape[1])))
            return Matrix(result, self.shape)
        return NotImplemented

    def __radd__(self, b):
        """
        TODO Vector return Vector
        """
        return self.__add__(b)

    def __sub__(self, b):
        if type(b) is int or type(b) is float:
            result = []
            for row in self.data:
                result.append(list(col - b for col in row))
            return Matrix(result, self.shape)
        elif isinstance(b, Matrix):
            if self.shape != b.shape:
                raise TypeError(
                    'You can only substract Matrices of the same size.')
            result = []
            for row in range(self.shape[0]):
                result.append(
                    list(self.data[row][col] + b.data[row][col] for col in range(self.shape[1])))
            return Matrix(result, self.shape)
        return NotImplemented

    def __rsub__(self, b):
        """
        TODO Vector return Vector
        """
        return self.__sub__(b)

    def __truediv__(self, b):
        if type(b) is int or type(b) is float:
            if int(b) == 0:
                raise ZeroDivisionError("You can't divide a Matrix by 0 !")
            result = []
            for row in self.data:
                result.append(list(col / b for col in row))
            return type(self)(result, self.shape)
        return NotImplemented

    def __rtruediv__(self, b):
        return NotImplemented

    def __mul__(self, b):
        """
        TODO Vector return Vector
        """
        if isinstance(b, Matrix):
            # m*n n*1 -> m*1
            if self.shape[1] == b.shape[0] and b.shape[1] == 1:
                result = []
                for row in self.data:
                    new_row = 0
                    for index in range(self.shape[1]):
                        new_row += (row[index] * b.data[index][0])
                    result.append([new_row])
                if isinstance(self, Vector) or isinstance(b, Vector):
                    return Vector(result)
                return Matrix(result, (self.shape[0], 1))
            # m*n n*p -> m*p
            elif self.shape[1] == b.shape[0]:
                result = []
                for row in range(self.shape[0]):
                    current_row = []
                    for index in range(b.shape[1]):
                        row_result = 0
                        for j in range(self.shape[1]):
                            row_result += (
                                self.data[row][j] * b.data[j][index])
                        current_row.append(row_result)
                    result.append(current_row)
                if isinstance(self, Vector) or isinstance(b, Vector):
                    return Vector(result)
                return Matrix(result, (self.shape[0], b.shape[1]))
            raise TypeError(
                "You can only multiply a Matrix with the same shape or a vector that match the rows.")
        elif type(b) is int or type(b) is float:
            result = []
            for row in self.data:
                result.append(list(col * b for col in row))
            return Matrix(result, self.shape)
        return NotImplemented

    def __rmul__(self, b):
        return self.__mul__(b)

    def __str__(self):
        return '(Matrix {})'.format(str(self.data))

    def __repr__(self):
        return '<Matrix of shape {} {}>'.format(self.shape, self.data)


class Vector(Matrix):
    def __init__(self, initializer: list, shape=None) -> None:
        if not isinstance(initializer, list):
            raise TypeError('Vector initialized must be a list.')
        is_1d = Vector.validate_list(initializer)
        # Wrap row vector inside a list to be a Matrix
        if is_1d:
            super().__init__([initializer], shape)
        else:
            super().__init__(initializer, shape)

    def validate_list(number_list: list,  allow_nested_list=True) -> bool:
        """
        Validate a list of float or nested list of floats against all errors.
        The number_list should strictly be a list of float,
        or a list of list float with the exact same length.
        """
        has_one_float = not allow_nested_list or len(number_list) == 0
        has_one_list = False
        has_empty_row = False
        for i in number_list:
            is_float = isinstance(i, float)
            is_list = isinstance(i, list)
            if has_empty_row:
                raise TypeError(
                    'A zero Vector can only have a single row/column')
            if (has_one_list and is_float) or (has_one_float and not is_float) or (is_list and not allow_nested_list):
                raise TypeError(
                    'Vector initializer must be an array of float or a nested array of float, got {}.'.format(i))
            if is_list:
                current_length = len(i)
                if current_length > 1:
                    raise TypeError(
                        'A Vector must have only one row or one column.')
                elif current_length == 0:
                    has_empty_row = True
                has_one_list = True
                Vector.validate_list(i, False)
            else:
                has_one_float = True
        return has_one_float

    def is_column_vector(self) -> bool:
        return len(self.values) > 0 and isinstance(self.values[0], list)

    def is_row_vector(self) -> bool:
        return not self.is_column_vector()

    def dot(self, b):
        if isinstance(b, Vector):
            if self.shape != b.shape:
                raise ArithmeticError(
                    "Two Vectors need the same dimension to be produce a dot product.")
            # Handle zero vector
            if self.shape == (0, 0):
                return 0.0
            # Since a matrix is used no need to check for Vector type
            return sum([a[0] * b[0] for (a, b) in zip(self.data, b.data)])
        return NotImplemented

    def __str__(self):
        return '(Vector {})'.format(str(self.data))

    def __repr__(self):
        return '<Vector of shape {} {}>'.format(self.shape, self.data)
