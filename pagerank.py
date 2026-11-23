import numpy as np
from scipy.sparse import csr_array

vecf64 = np.ndarray[tuple[int], np.dtype[np.float64]]

def mult_G(P: csr_array, alpha: int, v: vecf64, x: vecf64) -> vecf64:
    y = alpha * (P @ x)
    return y + (1 - y.sum()) * v

def pagerank(P: csr_array, alpha: int, tolerance: int) -> vecf64:
    shape = P.shape
    v = np.full(shape[0], 1 / shape[0])
    x = v.copy()
    
    while True:
        new_x = mult_G(P, alpha, v, x)
        
        if (np.linalg.norm(new_x - x) <= tolerance):
            return new_x

        x = new_x

def main():
    P = csr_array([[0.25, 0, 0.66],
                   [0.25, 0, 0],
                   [0.5, 0, 0.34]], dtype=np.float64)
    alpha = 0.8
    epsilon = 10e-9
    x = pagerank(P, alpha, epsilon)
    print(f'eigenvector = {x}')
    v = np.full(P.shape[0], 1 / P.shape[0])
    print(f'Gx = {mult_G(P, alpha, v, x)}')

if __name__ == '__main__':
    main()
