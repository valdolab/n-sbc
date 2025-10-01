import numpy as np


def hamming_distance(xi, xw):
    """
    Numpy version to calculate hamming distance
    """
    return np.sum(xi != xw)


def hamming_distance_opimized(xi, xw):
    """
    Most optimized for binary vectors using XOR
    Fastest but only for binary data
    """
    return np.sum(np.bitwise_xor(xi, xw))
