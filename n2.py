import time
import numpy as np


def matrix_separe(x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # dimensoes das matrizes
    n, m = x.shape
    n_half, m_half = n // 2, m // 2

    # divisao do problema em 4 subpartes
    x1 = x[:n_half, :m_half]
    x2 = x[:n_half, m_half:]
    x3 = x[n_half:, :m_half]
    x4 = x[n_half:, m_half:]
    return x1, x2, x3, x4


def matrix_multiply_n2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # caso base
    if len(x) == 1:
        return x * y

    a, b, c, d = matrix_separe(x)
    e, f, g, h = matrix_separe(y)

    p1 = matrix_multiply_n2(a, f - h)
    p2 = matrix_multiply_n2(a + b, h)
    p3 = matrix_multiply_n2(c + d, e)
    p4 = matrix_multiply_n2(d, g - e)
    p5 = matrix_multiply_n2(a + d, e + h)
    p6 = matrix_multiply_n2(b - d, g + h)
    p7 = matrix_multiply_n2(a - c, e + f)

    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    z = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

    return z


# dimensoes das matrizes
row_col_size = 2048
print("dimensoes: ", row_col_size)
start = time.time()
# quantidade de execucoes
for i in range(0, 1):
    x = np.random.rand(row_col_size, row_col_size)
    y = np.random.rand(row_col_size, row_col_size)
    r = matrix_multiply_n2(x, y)
end = time.time()
print(end - start)
