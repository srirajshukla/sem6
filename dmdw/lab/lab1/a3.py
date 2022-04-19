# Question 1
## a) Exp

from functools import lru_cache
import math


@lru_cache(maxsize=None)
def fac(n):
    if n == 0:
        return 1
    else:
        return n * fac(n-1)

def expn(x, n):
    e = 1
    for i in range(1, n):
        e += x**i / fac(i)
    return e

print(expn(0, 2))
print(expn(1, 10))
print(math.e)


## b) cos(x)
def cosine(x, n):
    c = 1
    for i in range(1, n):
        c += (-1)**i * x**(2*i) / fac(2*i)
    return c

print(cosine(math.pi/2, 3))


## c) 1/1-x
def inverse(x, n):
    s = 1
    for i in range(1, n):
        s += x**i
    return s

print(inverse(0.5, 3))

## d) ln(1+x)
def natural_log(x, n):
    l = 0
    for i in range(1, n+1):
        l += (-1)**(i+1) * x**i / i
    return l

print(natural_log(0.5, 2))

## e) tan inverse
def tan_inv(x, n):
    t = 0
    for i in range(1, n+1):
        t += (-1)**(i+1) * x**(2*i-1) / (2*i-1)
    return t

print(tan_inv(1, 3))

# question 2
## bisection method
## f(x) = sin(x) + 1/1-x

def bis_fun(x):
    return math.sin(x) + inverse(x, 8)

def bisect(a, b, eps):
    x0 = (a + b) / 2
    while bis_fun(x0) > eps:
        if bis_fun(a) * bis_fun(x0) < 0:
            b = x0
        else:
            a = x0
        x0 = (a + b) / 2