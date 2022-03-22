"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    """
    :math:`f(x, y) = x * y`

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        The product of x and y
    """
    return x * y


def id(x):
    """
    :math:`f(x) = x`

    Args:
        `x` (Any)

    Returns:
        `x` itself
    """
    return x


def add(x, y):
    """
    :math:`f(x, y) = x + y`

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        The sum of x and y
    """
    return x + y


def neg(x):
    """
    :math:`f(x) = -x`

    Args:
        x (scalar): The number to negate

    Returns:
        Negative `x`
    """
    return -x


def lt(x, y):
    """
    :math:`f(x) =` 1.0 if x is less than y else 0.0

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        1.0 if x is less than y else 0.0
    """
    return 1.0 if x < y else 0.0


def eq(x, y):
    """
    :math:`f(x) =` 1.0 if x is equal to y else 0.0

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        1.0 if x is equals y else 0.0
    """
    return 1.0 if x == y else 0.0


def max(x, y):
    """
    :math:`f(x) =` x if x is greater than y else y

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        x if x > y else y
    """
    return x if x > y else y


def is_close(x, y):
    """
    :math:`f(x) = |x - y| < 1e-2`

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        1.0 if x is less than y else 0.0
    """
    return abs(x - y) < 1e-2


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    return x if x > 0.0 else 0.0


EPS = 1e-6


def log(x):
    r"""
    :math:`f(x) = log(x)`

    Args:
        x (float): input

    Returns:
        float : natural log of input
    """
    return math.log(x + EPS)


def exp(x):
    r"""
    :math:`f(x) = e^{x}`

    Args:
        x (float): input

    Returns:
        float : exponential of value
    """
    return math.exp(x)


def log_back(x, d):
    r"""
    If :math:`f = log` as above, compute d :math:`d \times f'(x)`

    Args:
        x (float): input
        d (float): step size

    Returns:
        float : derivative of log times the step size
    """
    return d * (1 / (x + EPS))


def inv(x):
    """
    :math:`f(x) = 1/x`

    Args:
        x (float): input

    Returns:
        float : multiplicative inverse of x
    """
    return 1 / x


def inv_back(x, d):
    r"""
    If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`

    Args:
        x (float): input
        d (float): step size

    Returns:
        float : derivative of 1/x times the step size
    """
    return -d / x**2


def relu_back(x, d):
    r"""
    If :math:`f = relu` compute d :math:`d \times f'(x)`

    Args:
        x (float): input
        d (float): step size

    Returns:
        float : derivative of RELU times the step size
    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def mapped_fn(ls):
        return [fn(i) for i in ls]
    
    return mapped_fn


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    def zipped_fun(ls1, ls2):
        return [fn(a, b) for a, b in zip(ls1, ls2)]
    
    return zipped_fun


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def reduced_fn(ls):
        result = start
        for x in ls:
            result = fn(result, x)
        
        return result

    return reduced_fn


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0)(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1)(ls)
