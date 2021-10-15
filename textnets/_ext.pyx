# compiled function for the integrand of the disparity filter for significant speedup
cdef double df_integrand(int n, double[2] args) nogil:

    x = args[0]
    degree = args[1]

    return (1 - x) ** (degree - 2)
