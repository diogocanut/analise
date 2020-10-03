import numpy as np
import time

def matrix_multiply_n3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # n x m e m x p
    n, m = x.shape
    m, p = y.shape

    z = np.zeros((n, p))
    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, m):
                z[i][j] = z[i][j] + (x[i][k] * y[k][j])

    return z

# dimensoes das matrizes
row_col_size = 2048

print("dimensoes: ", row_col_size)
start = time.time()
for i in range(0, 1):
    x = np.random.rand(row_col_size, row_col_size)
    y = np.random.rand(row_col_size, row_col_size)
    r = matrix_multiply_n3(x, y)
end = time.time()
print(end - start)
