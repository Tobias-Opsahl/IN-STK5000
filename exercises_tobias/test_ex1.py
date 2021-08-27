import unittest

import sys
sys.path.insert(0, "C:/Users/tobia/Documents/Host2021/IN-STK5000/IN-STK5000/exercises_tobias")
import ex1 as ex1_solution

class TestEx1(unittest.TestCase):

    def test_task1(self):
        l = list('abcd')
        self.assertEqual(ex1_solution.task1(l, 2), ['a', 'b'])
        self.assertEqual(ex1_solution.task1(l, -2), ['c', 'd'])

    def test_task2(self):

        l = list("abc")
        for n, item in enumerate(l):
            self.assertEqual(ex1_solution.task2(l, n), item)
        self.assertEqual(ex1_solution.task2(l, len(l)), None)

    def test_task3(self):

        self.assertEqual(ex1_solution.task3(list("abcd")),
                        list("dcba"))

    def test_task4(self):

        self.assertEqual(ex1_solution.task4(range(5), 3),
                         [0, 1, 8, 27, 64])

    def test_task5(self):

        fib = ex1_solution.Fibonacci(1, 1)
        seq = [fib.next() for _ in range(7)]
        self.assertEqual(seq, [1, 1, 2, 3, 5, 8, 13])

    def test_task6(self):

        from itertools import islice

        self.assertEqual(list(islice(ex1_solution.fibonacci(1, 1), 7)),
            [1,1,2,3,5,8,13])

if __name__ == "__main__":
    unittest.main()
