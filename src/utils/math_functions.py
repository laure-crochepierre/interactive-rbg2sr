import numpy as np
np.seterr(all="raise")


def minimum(list_of_arrays):
    x = np.array(list_of_arrays)
    return x.min(-2)


def maximum(list_of_arrays):
    x = np.array(list_of_arrays)
    return x.max(-2)


def abs_list(li) :
    li = np.abs(np.array(li))
    return [el for el in abs_list(li)]


def neg(A):
    return np.where(A > 0, A, 0)


def pos(A):
    return np.where(A < 0, A, 0)


def opposite(A):
    return -A


def sqrt_add(A, B):
    return np.sqrt(np.add(A, B))


def if_then_else(A, B, C):
    return np.where(A, B, C)
