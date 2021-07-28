

def task1(input_list, n):
    """Return the n first elements of a list if n>0, else th n last elements."""
    if n > 0:
        return input_list[:n]
    else:
        return input_list[n:]


def task2(input_list, n):
    """Return the n-th element of a list, or None if the list is shorter than n
    elements."""

    if len(input_list) <= n:
        return None
    return input_list[n]


def task3(input_list):
    """Return the reverse of a given list."""

    return input_list[::-1]


def task4(input_sequence, p):
    """Raise the elements of the input sequence to the p-th power."""

    return [i**p for i in input_sequence]


class Fibonacci:

    """Create a class that creates a Fibonacci sequence. The first two elements
    are given in the constructor. The next function should return at each call
    the next number in the sequence, starting with the two elemens passed to the
    constructor."""

    def __init__(self, first, second):

        self.first = first
        self.second = second

    def next(self):

        res = self.first
        self.first, self.second = self.second, self.first+self.second
        return res


def fibonacci(a, b):
    """Create a generator function that generates an infinite Fibonacci 
    sequence, just like the Fibonacci class starting with a and b."""

    while True:
        yield a
        a, b = b, a+b
