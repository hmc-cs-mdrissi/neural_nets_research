#
# A simple test
#

def fib(n):
    if n == 0 or n == 1:
        return 1
    return fib(n-1) + fib(n-2)

def test_fib():
    assert fib(4) == 5
