#!/usr/bin/env python2
# coding: utf-8

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

    def __len__(self):
        return len(self.data)

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

    def __mul__(self, number):
        """ overload * operator, multiplied by a number """
        return Vector([self.data[i] * number for i in xrange(len(self))])

    def __imul__(self, number):
        """ overload *= operator, multiplied by a number """
        for i in xrange(len(self)):
            self.data[i] *= number

        return self

    @classmethod
    def dot_prod(cls, one, another):
        if one.__class__.__name__ != 'Vector' or one.__class__.__name__ != another.__class__.__name__:
            raise TypeError('Type is not Vector')

        if len(one) != len(another):
            raise ValueError('Unequal length of vectors')

        return sum(one.data[i] * another.data[i] for i in xrange(len(one)))

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
        return Matrix(self.row_num, self.col_num, self.data)

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

    m = Matrix(3, 3, data = [1, 2, 3, 4, 5, 6, 7, 8, 9])
    m2 = Matrix(3, 2, data = [1, 2, 2, 1, 2, 1])
    print m.data
    print m2.data
    print mmul(m, m2).data


