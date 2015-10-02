#!/usr/bin/env python2
# coding: utf-8

import random, copy

class Vector:
    """
    Vector class uses python list as its internal structure.
    
    Implemented operations like vector plus, multiplication by a number and 
    dot product.
    """
    def __init__(self, datalist):
        self.data = datalist

    @classmethod
    def fromList(cls, datalist):
        return Vector(datalist)

    @classmethod
    def fromIterable(cls, iterator):
        return Vector(list(iterator))

    @classmethod
    def fromRandom(cls, length):
        """ generate a vector of given length with all random float num in [0, 1) """
        return Vector([random.random() for i in xrange(length)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        return self

    def __eq__(self, other):
        return self.equalTo(other, 0)

    def __ne__(self, other):
        return self.notEqualTo(other, 0)

    def equalTo(self, other, threshold = 0):
        return (len(self) == len(other)
                and all(abs(self.data[i] - other.data[i]) <= threshold
                    for i in xrange(len(self.data))))

    def notEqualTo(self, other, threshold = 0):
        return (len(self) != len(other)
                or any(abs(self.data[i] - other.data[i]) > threshold
                    for i in xrange(len(self.data))))

    def __add__(self, another):
        """ overload + operator """
        if another.__class__.__name__ != self.__class__.__name__:
            raise TypeError('Another is not Vector')

        if len(self) != len(another):
            raise ValueError('Unequal length of vectors')

        return Vector([self.data[i] + another.data[i] for i in xrange(len(self))])

    def __iadd__(self, another):
        """ overload += operator """
        if another.__class__.__name__ != self.__class__.__name__:
            raise TypeError('Another is not Vector')

        if len(self) != len(another):
            raise ValueError('Unequal length of vectors')

        for i in xrange(len(self)):
            self.data[i] += another.data[i]

        return self

    def __sub__(self, another):
        """ overload - operator """
        if another.__class__.__name__ != self.__class__.__name__:
            raise TypeError('Another is not Vector')

        if len(self) != len(another):
            raise ValueError('Unequal length of vectors')

        return Vector([self.data[i] - another.data[i] for i in xrange(len(self))])

    def __isub__(self, another):
        """ overload -= operator """
        if another.__class__.__name__ != self.__class__.__name__:
            raise TypeError('Another is not Vector')

        if len(self) != len(another):
            raise ValueError('Unequal length of vectors')

        for i in xrange(len(self)):
            self.data[i] -= another.data[i]

        return self

    def __mul__(self, number):
        """ overload * operator, multiplied by a number """
        return Vector([self.data[i] * number for i in xrange(len(self))])

    def __rmul__(self, number):
        """ overload * operator, multiplied by a number on the left """
        return self.__mul__(number)

    def __imul__(self, number):
        """ overload *= operator, multiplied by a number """
        for i in xrange(len(self)):
            self.data[i] *= number

        return self

    def append(self, num):
        self.data.append(num)
        return self

    def extend(self, iterator):
        self.data.extend(iterator)
        return self

    @classmethod
    def dot_prod(cls, one, another):
        if one.__class__.__name__ != 'Vector' or one.__class__.__name__ != another.__class__.__name__:
            raise TypeError('Type is not Vector')

        if len(one) != len(another):
            raise ValueError('Unequal length of vectors')

        return sum(one.data[i] * another.data[i] for i in xrange(len(one)))

    def __copy__(self):
        return Vector(copy.copy(self.data))

    def assign(self, vec):
        '''
        assign values to a vector, in order to avoid to copy a large object
        '''
        if len(self) != len(vec):
            raise ValueError('Unequal length of vectors')

        for i in xrange(len(vec)):
            self.data[i] = vec[i]

def dot_prod(one, another):
    return Vector.dot_prod(one, another)

class Matrix:
    """
    Matrix uses a single list as its internal structure
    By default it will be initialized as a matrix full of zeros
    """
    row_num = 0
    col_num = 0
    data = []

    def __init__(self, row_num, col_num, data = None, bycol = False):
        """
        """
        if bycol:
            raise NotImplementedError('init by row only for now')

        if data is None:
            self.data = [ 0 for i in xrange(row_num * col_num) ] # use one line list to store
        else:
            if len(data) != row_num * col_num:
                raise ValueError('input data does not fit the desired matrix')

            self.data = data

        self.row_num = row_num
        self.col_num = col_num

    @classmethod
    def fromRandom(cls, row_num, col_num):
        """
        Generate a matrix with given column and row number,
        which is filled with all random floating numbers from [0, 1)
        """
        return Matrix(row_num, col_num,
                [random.random() for i in xrange(row_num * col_num) ])

    @classmethod
    def fromIterable(self, row_num, col_num, dataiter, bycol = False):
        return Matrix(row_num, col_num, list(dataiter))

    def __eq__(self, other):
        return self.equalTo(other, 0)

    def __ne__(self, other):
        return self.notEqualTo(other, 0)

    def equalTo(self, other, threshold = 0):
        return (self.row_num == other.row_num and self.col_num == other.col_num
                and all(abs(self.data[i] - other.data[i]) <= threshold
                    for i in xrange(len(self.data))))

    def notEqualTo(self, other, threshold = 0):
        return (self.row_num != other.row_num or self.col_num != other.col_num
                or any(abs(self.data[i] - other.data[i]) > threshold
                    for i in xrange(len(self.data))))

    def __add__(self, other):
        return Matrix(self.row_num, self.col_num, data = list(
            self.data[i] + other.data[i] for i in xrange(len(self.data))
            ))

    def __iadd__(self, other):
        if self.row_num != other.row_num or other.col_num != other.col_num:
            raise ValueError('two matrices are not in the same size')

        for i in xrange(len(self.data)):
            self.data[i] += other.data[i]

        return self

    def __sub__(self, other):
        return Matrix(self.row_num, self.col_num, data = list(
            self.data[i] - other.data[i] for i in xrange(len(self.data))
            ))

    def __isub__(self, other):
        if self.row_num != other.row_num or other.col_num != other.col_num:
            raise ValueError('two matrices are not in the same size')

        for i in xrange(len(self.data)):
            self.data[i] -= other.data[i]

        return self

    def item(self, row_id, col_id):
        if row_id < 0 or self.row_num <= row_id:
            raise ValueError('Not a valid row id')

        if col_id < 0 or self.col_num <= col_id:
            raise ValueError('Not a valid col id')

        return self.data[row_id * self.col_num + col_id]

    def set(self, row_id, col_id, val):
        if row_id < 0 or self.row_num <= row_id:
            raise ValueError('Not a valid row id')

        if col_id < 0 or self.col_num <= col_id:
            raise ValueError('Not a valid col id')

        self.data[row_id * self.col_num + col_id] = val

        return self

    def row(self, row_id):
        """
        Return the row vector, given a zero-based row id
        """
        if row_id < 0 or self.row_num <= row_id:
            raise ValueError('Not a valid row id')

        return Vector.fromIterable(self.data[i] for i in xrange(row_id * self.col_num, (row_id + 1) * self.col_num))

    def col(self, col_id):
        """
        Return the column vector, given a zero-based column id
        """
        if col_id < 0 or self.col_num <= col_id:
            raise ValueError('Not a valid col id')

        return Vector.fromIterable(self.data[row_id * self.col_num + col_id] for row_id in xrange(self.row_num))

    def __mul__(self, number):
        """ overload * operator, multiplied by a number """
        return Matrix(self.row_num, self.col_num, [x * number for x in self.data])

    def __rmul__(self, number):
        """ overload * operator, multiplied by a number """
        return self.__mul__(number)

    def __imul__(self, number):
        """ overload *= operator, multiplied by a number """
        for i in self.col_num * self.row_num:
            self.data[i] *= number

        return self

    @classmethod
    def mul(cls, one, other):
        if one.__class__.__name__ != 'Matrix' or one.__class__.__name__ != other.__class__.__name__:
            raise TypeError('Both objects should be Matrix')

        if one.col_num != other.row_num:
            raise ValueError('Unequal column number and row number')

        return Matrix(one.row_num, other.col_num,
                list(dot_prod(one.row(row_id), other.col(col_id))
                    for row_id in xrange(one.row_num)
                    for col_id in xrange(other.col_num)))

def mmul(one, another):
    return Matrix.mul(one, another)

def vmul(m, vec):
    """
    Multiply a matrix with a vector on the right, output a vector again
    """
    if m.__class__.__name__ != 'Matrix' or vec.__class__.__name__ != 'Vector':
        raise TypeError('Matrix and Vector is required to do multiplication')

    if m.col_num != len(vec):
        raise ValueError('Unequal column number and vector length')

    return Vector.fromIterable(dot_prod(m.row(row_id), vec)
            for row_id in xrange(m.row_num))

if __name__ == "__main__":
    v = Vector([1, 2, 3])
    print v.data

    v = Vector.fromList([4, 5, 6])
    print v.data

    v = Vector.fromIterable(xrange(10, 15))
    print v.data

    v2 = Vector.fromIterable(xrange(5))
    v3 = v + v2

    print v3.data

    v = Vector.fromIterable(xrange(4))
    v2 = Vector.fromIterable(xrange(4, 0, -1))
    print dot_prod(v, v2)

    v = Vector.fromRandom(3)
    print v.data

    v2 = copy.copy(v)
    v2[0] = 3.1415926
    print v.data
    print v2.data

    print "-------------"

    m = Matrix(3, 3)
    print m.data

    m = Matrix(3, 3, data = list(xrange(1, 10)))
    print m.data

    print m.item(1, 1)
    m.set(1, 1, 100)
    print m.item(1, 1)

    v = m.row(1)

    print m.row(1).data
    print m.col(1).data

    print "mmul"
    m = Matrix(3, 3, data = [1, 2, 3, 4, 5, 6, 7, 8, 9])
    m2 = Matrix(3, 2, data = [1, 2, 2, 1, 2, 1])
    print m.data
    print m2.data
    print mmul(m, m2).data

    print "Matrix.fromRandom"
    m = Matrix.fromRandom(3, 4)
    print m.data

    print "vmul"
    m = Matrix(3, 2, data = [1, 2, 3, 4, 5, 6])
    v = Vector([1, 10])
    print vmul(m, v).data

    print "Matrix.fromIterable"
    m = Matrix.fromIterable(3, 3, xrange(9))
    print m.data

    print "Matrix add + "
    m = Matrix.fromIterable(3, 2, xrange(6))
    m2 = Matrix.fromIterable(3, 2, (i * 20 for i in xrange(6)))
    print (m + m2).data
    print m.data
    m += m2
    print m.data

    print "Matrix sub - "
    m = Matrix.fromIterable(3, 2, xrange(6))
    m2 = Matrix(3, 2, [1] * 6)
    print (m - m2).data
    print m.data
    m -= m2
    print m.data



