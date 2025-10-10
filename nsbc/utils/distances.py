import numpy as np
from numba import jit, prange


def hamming_distance(xi, xw):
    """
    Most optimized for binary vectors using XOR
    Fastest but only for binary data. Faster than using np.sum()
    """
    return np.count_nonzero(xi != xw)


def hamming_distance_packed(xi, xw):
    """
    For very long binary vectors, pack bits into integers
    32x memory reduction and faster processing
    """
    xi_packed = np.packbits(xi.astype(np.uint8))
    xw_packed = np.packbits(xw.astype(np.uint8))

    xor_result = np.bitwise_xor(xi_packed, xw_packed)
    return np.sum(np.unpackbits(xor_result)[: len(xi)])


@jit(nopython=True, parallel=True)
def hamming_batch_bitpacked_parallel(x_packed, y_packed):
    """
    Parallel JIT-compiled - FASTEST BIT-PACKED VERSION

    Uses all CPU cores for maximum speed
    """
    n_samples = x_packed.shape[0]
    m_samples = y_packed.shape[0]
    distances = np.zeros((n_samples, m_samples), dtype=np.int32)

    for i in prange(n_samples):
        for j in range(m_samples):
            count = 0
            for k in range(x_packed.shape[1]):
                xor_byte = x_packed[i, k] ^ y_packed[j, k]
                while xor_byte:
                    xor_byte &= xor_byte - 1
                    count += 1

            distances[i, j] = count
    return distances


def hamming_bitpacked_parallel(x, y):
    """
    Best bit-packed method for big data and avoid overflow memory issue

    Usage:
        X = np.random.randint(0, 2, (50000, 1000), dtype=np.uint8)
        Y = np.random.randint(0, 2, (10000, 1000), dtype=np.uint8)
        distances = hamming_bitpacked_parallel(X, Y)
    """
    x_packed = np.packbits(x, axis=1)
    y_packed = np.packbits(y, axis=1)
    return hamming_batch_bitpacked_parallel(x_packed, y_packed)
