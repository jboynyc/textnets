cdef double integrand(int n, double[2] args):

    x = args[0]
    degree = args[1]

    return (1 - x) ** (degree - 2)
