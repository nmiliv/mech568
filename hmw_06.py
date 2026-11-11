# AI use statement: AI was used to convert matlab code to python.

## imports
import time
import numpy as np
from scipy.io import loadmat
import scipy.sparse
from scipy.sparse import diags
import warnings
import matplotlib.pyplot as plt

demo = True
LU = True
jacobis = True
gauss_seidels = True
SORs = True
conj_grads = True
pre_conj_grads = True
sors_investigate = True
part2 = True
testing = False

def LU_decomp_solve(A, b):
    n = A.shape[0]
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    # Pivoting
    P = np.eye(n)
    for i in range(n - 1):
        row_num_small = np.argmax(np.abs(A[i:, i]))
        row_num = i + row_num_small
        P[[i, row_num], :] = P[[row_num, i], :]

    # LU Decomposition
    U = P @ A
    L = np.eye(n)
    for i in range(n - 1):
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]

    # Forward substitution (Ld = Pb)
    d = np.zeros(n)
    b_perm = P @ b
    for i in range(n):
        sum_val = np.dot(L[i, :i], d[:i]) if i > 0 else 0.0
        d[i] = b_perm[i] - sum_val

    # Backward substitution (Ux = d)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = np.dot(U[i, i:], x[i:])
        x[i] = (d[i] - sum_val) / U[i, i]

    return x

def Precond_Conj_Grad(A, b, M, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    M_inv = np.linalg.inv(M)
    r = b - np.dot(A, x) # TODO not sure this dot product is correct
    s = M_inv @ r
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        alpha = (r @ M_inv @ r) / (s @ A @ s) # todo should we be transposing these?
        x_new = x + alpha * s
        r_new = r - alpha * (A @ s)
        beta = (r_new @ M_inv @ r_new) / (r @ r)
        s = (M_inv @ r_new) + beta * s

        err = np.linalg.norm(x_new - x)
        x = x_new
        r = r_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')

    return x, iter_count

def Conj_Grad(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)
    s = r.copy()
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        alpha = np.dot(r, r) / (s @ A @ s)
        x_new = x + alpha * s
        r_new = r - alpha * (A @ s)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        s = r_new + beta * s

        err = np.linalg.norm(x_new - x)
        x = x_new
        r = r_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')

    return x, iter_count

def SOR(A, b, omega, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        x_new = np.zeros(n)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i]) if i > 0 else 0.0
            sum2 = np.dot(A[i, i+1:], x[i+1:]) if i < n - 1 else 0.0

            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]

        err = np.linalg.norm(x_new - x)
        x = x_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')

    return x, iter_count

def gauss_seidel(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        x_new = np.zeros(n)
        for i in range(n):
            temp_sum = 0.0
            for j in range(n):
                if j > i:
                    temp_sum += A[i, j] * x[j]
                elif j < i:
                    temp_sum += A[i, j] * x_new[j]
            x_new[i] = (b[i] - temp_sum) / A[i, i]

        err = np.linalg.norm(x_new - x)
        x = x_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')

    return x, iter_count

def jacobi(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        x_new = np.zeros(n)
        for i in range(n):
            temp_sum = 0.0
            for j in range(n):
                if j != i:
                    temp_sum += A[i, j] * x[j]
            x_new[i] = (b[i] - temp_sum) / A[i, i]

        err = np.linalg.norm(x_new - x)
        x = x_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')

    return x, iter_count

def magic(n):
    """Generates a magic square of order n (supports odd and doubly-even n)."""
    if n < 3:
        raise ValueError("Magic square size must be >= 3")
    if n % 2 == 1:
        p = np.arange(1, n+1)
        i = (p%n).astype(int)
        j = ((p*2 - 1)%n).astype(int)
        J = np.outer(np.ones(n), j) + np.outer(i, np.zeros(n))
        I = np.outer(i, np.ones(n)) + np.outer(np.zeros(n), j)
        M = np.arange(1, n*n + 1).reshape(n, n)
        return M[I.astype(int), J.astype(int)]
    elif n % 4 == 0:
        M = np.arange(1, n*n + 1).reshape(n, n)
        I = (M % 4 == 1) | (M % 4 == 2)
        M[I] = n*n + 1 - M[I]
        return M
    else:
        # Singly even order
        p = n // 2
        sub = magic(p)
        M = np.block([[sub, sub + 2*p*p], [sub + 3*p*p, sub + p*p]])
        idx = np.arange(p)
        k = (n - 2) // 4
        j = np.concatenate((np.arange(k), np.arange(n - k + 1, n)))
        for i in range(p):
            for col in j:
                M[[i, i + p], col] = M[[i + p, i], col]
        col = k
        M[[k, k + p], [0, k]] = M[[k + p, k], [0, k]]
        for i in range(k + 1, k + p + 1):
            if i != k + 1:
                M[[i, i + p], col] = M[[i + p, i], col]
        return M

# Parameters
n = 1000
tol = 1e-10
max_iter = 1000 # TODO change this back to 1000
omeg = 1.3

# System 1
A1 = magic(n).astype(float)
A1 = A1.T @ A1
b1 = np.ones(n)
D1 = np.diag(np.diag(A1))
M1 = D1
if demo: print(f"rcond(A1) = {1.0 / np.linalg.cond(A1):e}")

# System 2 (Load from .mat files using scipy)
data_A2 = loadmat('A2.mat')
A2 = data_A2['B'].astype(float)
data_b2 = loadmat('b2.mat')
b2 = data_b2['BC_vec'].flatten().astype(float)
D2 = np.diag(np.diag(A2))
M2 = D2
if demo: print(f"rcond(A2) = {1.0 / np.linalg.cond(A2):e}")

# System 3
n3 = 2500
diagonals = [-1 * np.ones(n3), 4 * np.ones(n3), -1 * np.ones(n3)]
A3 = diags(diagonals, [-1, 0, 1], shape=(n3, n3)).toarray()
b3 = np.ones(n3)
D3 = np.diag(np.diag(A3))
M3 = D3
if demo: print(f"rcond(A3) = {1.0 / np.linalg.cond(A3):e}")

# System 4
diagonals = [1 * np.ones(n3), 0 * np.ones(n3), -1 * np.ones(n3)]
A4 = diags(diagonals, [-1, 0, 1], shape=(n3, n3)).toarray()
b4 = np.ones(n3)
A4 = A4.T @ A4
b4 = A4.T @ b4
if demo: print(f"rcond(A4) = {1.0 / np.linalg.cond(A4):e}")

if LU:
    # Solve and display results for System 1 using LU Decomp
    start_time = time.time()
    x1_LU = LU_decomp_solve(A1, b1)
    time1_lu = time.time() - start_time
    r_lu1 = np.linalg.norm(A1 @ x1_LU - b1)

    print('--- Results for System 1 LU (A1, b1) ---')
    print(f'LU Decomp: Time = {time1_lu:.4f} sec, Residual = {r_lu1:.4e}')

    start_time = time.time()
    x2_LU = LU_decomp_solve(A2, b2)
    time2_lu = time.time() - start_time
    r_lu2 = np.linalg.norm(A2 @ x2_LU - b2)

    print('--- Results for System 2 LU (A2, b2) ---')
    print(f'LU Decomp: Time = {time2_lu:.4f} sec, Residual = {r_lu2:.4e}')

    start_time = time.time()
    x3_LU = LU_decomp_solve(A3, b3)
    time3_lu = time.time() - start_time
    r_lu3 = np.linalg.norm(A3 @ x3_LU - b3)

    print('--- Results for System 3 LU (A3, b3) ---')
    print(f'LU Decomp: Time = {time3_lu:.4f} sec, Residual = {r_lu3:.4e}')

    start_time = time.time()
    x4_LU = LU_decomp_solve(A4, b4)
    time4_lu = time.time() - start_time
    r_lu4 = np.linalg.norm(A4 @ x4_LU - b4)

    print('--- Results for System 4 LU (A4, b4) ---')
    print(f'LU Decomp: Time = {time4_lu:.4f} sec, Residual = {r_lu4:.4e}')

if jacobis:
    # jacobi
    start_time = time.time()
    x1_LU, _ = jacobi(A1, b1, 1e-6, max_iter)
    time1_lu = time.time() - start_time
    r_lu1 = np.linalg.norm(A1 @ x1_LU - b1)

    print('--- Results for System 1 Jacobi (A1, b1) ---')
    print(f'Jacobi: Time = {time1_lu:.4f} sec, Residual = {r_lu1:.4e}')

    start_time = time.time()
    x2_LU, _ = jacobi(A2, b2, 1e-6, max_iter)
    time2_lu = time.time() - start_time
    r_lu2 = np.linalg.norm(A2 @ x2_LU - b2)

    print('--- Results for System 2 Jacobi (A2, b2) ---')
    print(f'Jacobi: Time = {time2_lu:.4f} sec, Residual = {r_lu2:.4e}')

    start_time = time.time()
    x3_LU, _ = jacobi(A3, b3, 1e-6, max_iter)
    time3_lu = time.time() - start_time
    r_lu3 = np.linalg.norm(A3 @ x3_LU - b3)

    print('--- Results for System 3 Jacobi (A3, b3) ---')
    print(f'Jacobi: Time = {time3_lu:.4f} sec, Residual = {r_lu3:.4e}')

    start_time = time.time()
    x4_LU, _ = jacobi(A4, b4, 1e-6, max_iter)
    time4_lu = time.time() - start_time
    r_lu4 = np.linalg.norm(A4 @ x4_LU - b4)

    print('--- Results for System 4 Jacobi (A4, b4) ---')
    print(f'Jacobi: Time = {time4_lu:.4f} sec, Residual = {r_lu4:.4e}')

if gauss_seidels:
    # gauss-seidel
    start_time = time.time()
    x1_LU, _ = gauss_seidel(A1, b1, 1e-6, max_iter)
    time1_lu = time.time() - start_time
    r_lu1 = np.linalg.norm(A1 @ x1_LU - b1)

    print('--- Results for System 1 Gauss-Seidel (A1, b1) ---')
    print(f'Gauss Seidel: Time = {time1_lu:.4f} sec, Residual = {r_lu1:.4e}')

    start_time = time.time()
    x2_LU, _ = gauss_seidel(A2, b2, 1e-6, max_iter)
    time2_lu = time.time() - start_time
    r_lu2 = np.linalg.norm(A2 @ x2_LU - b2)

    print('--- Results for System 2 Gauss-Seidel (A2, b2) ---')
    print(f'Gauss Seidel: Time = {time2_lu:.4f} sec, Residual = {r_lu2:.4e}')

    start_time = time.time()
    x3_LU, _ = gauss_seidel(A3, b3, 1e-6, max_iter)
    time3_lu = time.time() - start_time
    r_lu3 = np.linalg.norm(A3 @ x3_LU - b3)

    print('--- Results for System 3 Gauss-Seidel (A3, b3) ---')
    print(f'Gauss Seidel: Time = {time3_lu:.4f} sec, Residual = {r_lu3:.4e}')

    start_time = time.time()
    x4_LU, _ = gauss_seidel(A4, b4, 1e-6, max_iter)
    time4_lu = time.time() - start_time
    r_lu4 = np.linalg.norm(A4 @ x4_LU - b4)

    print('--- Results for System 4 Gauss-Seidel (A4, b4) ---')
    print(f'Gauss Seidel: Time = {time4_lu:.4f} sec, Residual = {r_lu4:.4e}')

if SORs:
    # SOR
    start_time = time.time()
    x1_LU, _ = SOR(A1, b1, omeg, 1e-6, max_iter)
    time1_lu = time.time() - start_time
    r_lu1 = np.linalg.norm(A1 @ x1_LU - b1)

    print('--- Results for System 1 SOR (A1, b1) ---')
    print(f'SOR: Time = {time1_lu:.4f} sec, Residual = {r_lu1:.4e}')

    start_time = time.time()
    x2_LU, _ = SOR(A2, b2, omeg, 1e-6, max_iter)
    time2_lu = time.time() - start_time
    r_lu2 = np.linalg.norm(A2 @ x2_LU - b2)

    print('--- Results for System 2 SOR (A2, b2) ---')
    print(f'SOR: Time = {time2_lu:.4f} sec, Residual = {r_lu2:.4e}')

    start_time = time.time()
    x3_LU, _ = SOR(A3, b3, omeg, 1e-6, max_iter)
    time3_lu = time.time() - start_time
    r_lu3 = np.linalg.norm(A3 @ x3_LU - b3)

    print('--- Results for System 3 SOR (A3, b3) ---')
    print(f'SOR: Time = {time3_lu:.4f} sec, Residual = {r_lu3:.4e}')

    start_time = time.time()
    x4_LU, _ = SOR(A4, b4, omeg, 1e-6, max_iter)
    time4_lu = time.time() - start_time
    r_lu4 = np.linalg.norm(A4 @ x4_LU - b4)

    print('--- Results for System 4 SOR (A4, b4) ---')
    print(f'SOR: Time = {time4_lu:.4f} sec, Residual = {r_lu4:.4e}')

if conj_grads:
    # conjugate gradient
    start_time = time.time()
    x1_LU, _ = Conj_Grad(A1, b1, 1e-6, max_iter)
    time1_lu = time.time() - start_time
    r_lu1 = np.linalg.norm(A1 @ x1_LU - b1)

    print('--- Results for System 1 Conjugate Gradient (A1, b1) ---')
    print(f'Conjugate Gradient: Time = {time1_lu:.4f} sec, Residual = {r_lu1:.4e}')

    start_time = time.time()
    x2_LU, _ = Conj_Grad(A2, b2, 1e-6, max_iter)
    time2_lu = time.time() - start_time
    r_lu2 = np.linalg.norm(A2 @ x2_LU - b2)

    print('--- Results for System 2 Conjugate Gradient (A2, b2) ---')
    print(f'Conjugate Gradient: Time = {time2_lu:.4f} sec, Residual = {r_lu2:.4e}')

    start_time = time.time()
    x3_LU, _ = Conj_Grad(A3, b3, 1e-6, max_iter)
    time3_lu = time.time() - start_time
    r_lu3 = np.linalg.norm(A3 @ x3_LU - b3)

    print('--- Results for System 3 Conjugate Gradient (A3, b3) ---')
    print(f'Conjugate Gradient: Time = {time3_lu:.4f} sec, Residual = {r_lu3:.4e}')

    start_time = time.time()
    x4_LU, _ = Conj_Grad(A4, b4, 1e-6, max_iter)
    time4_lu = time.time() - start_time
    r_lu4 = np.linalg.norm(A4 @ x4_LU - b4)

    print('--- Results for System 4 Conjugate Gradient (A4, b4) ---')
    print(f'Conjugate Gradient: Time = {time4_lu:.4f} sec, Residual = {r_lu4:.4e}')

if pre_conj_grads:
    # preconditioned conjugate gradient
    start_time = time.time()
    x1_LU, _ = Precond_Conj_Grad(A1, b1, M1, 1e-6, max_iter)
    time1_lu = time.time() - start_time
    r_lu1 = np.linalg.norm(A1 @ x1_LU - b1)

    print('--- Results for System 1 Preconditioned Conjugate Gradient (A1, b1) ---')
    print(f'Preconditioned Conjugate Gradient: Time = {time1_lu:.4f} sec, Residual = {r_lu1:.4e}')

    start_time = time.time()
    x2_LU, _ = Precond_Conj_Grad(A2, b2, M2, 1e-6, max_iter)
    time2_lu = time.time() - start_time
    r_lu2 = np.linalg.norm(A2 @ x2_LU - b2)

    print('--- Results for System 2 Preconditioned Conjugate Gradient (A2, b2) ---')
    print(f'Preconditioned Conjugate Gradient: Time = {time2_lu:.4f} sec, Residual = {r_lu2:.4e}')

    start_time = time.time()
    x3_LU, _ = Precond_Conj_Grad(A3, b3, M3, 1e-6, max_iter)
    time3_lu = time.time() - start_time
    r_lu3 = np.linalg.norm(A3 @ x3_LU - b3)

    print('--- Results for System 3 Preconditioned Conjugate Gradient (A3, b3) ---')
    print(f'Preconditioned Conjugate Gradient: Time = {time3_lu:.4f} sec, Residual = {r_lu3:.4e}')

    start_time = time.time()
    x4_LU, _ = Precond_Conj_Grad(A4, b4, -A4, 1e-6, max_iter)
    time4_lu = time.time() - start_time
    r_lu4 = np.linalg.norm(A4 @ x4_LU - b4)

    print('--- Results for System 4 Preconditioned Conjugate Gradient (A4, b4) ---')
    print(f'Preconditioned Conjugate Gradient: Time = {time4_lu:.4f} sec, Residual = {r_lu4:.4e}')


if sors_investigate:
    # effects of changing omega in SOR
    omegaspace = np.linspace(1.01, 1.99, num=20)
    resids = []
    iters = []
    for omega in omegaspace:
        x_omega, iter = SOR(A2, b2, omega, 1e-10, max_iter)
        r_x_omega = np.linalg.norm(A2 @ x_omega - b2)
        resids.append(r_x_omega)
        iters.append(iter)
    fig, ax = plt.subplots(1)
    ax.semilogy(omegaspace, resids)
    ax.set_xlabel("SOR omega")
    ax.set_ylabel("residuals")
    ax.set_title("Effect of omega on residual value in SOR")
    fig, ax = plt.subplots(1)
    ax.semilogy(omegaspace, iters)
    ax.set_xlabel("SOR omega")
    ax.set_ylabel("Iterations")
    ax.set_title("Effect of omega on convergence speed in SOR")

def make_A(n):
    A = scipy.sparse.diags_array([-1, 16, -30, 16, -1], offsets=[-2, -1, 0, 1, 2], shape=(n, n), dtype='float64')/12
    return A.toarray()

def solve_pent_LU(A, b):
    n = A.shape[0]
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    dm2 = np.ones(n-2) * A[0,2]
    dm1 = np.ones(n-1) * A[0,1]
    d0 = np.ones(n) * A[0,0]
    dp1 = np.ones(n-1) * A[0,1]
    dp2 = np.ones(n-2) * A[0,2]
    L1 = []
    L2 = []

    for i in range(n-2):
        L1.append(dm1[i]/d0[i])
        # dm1[0] -= dm1[0]/d0[0] * d0[0]
        d0[i+1] -= dm1[i]/d0[i] * dp1[i]
        dp1[i+1] -= dm1[i]/d0[i] * dp2[i]
        # dp2[1] -= dm1[0]/d0[0] * 0

        L2.append(dm2[i]/d0[i])
        # dm2[0] -= dm2[0]/d0[0] * d0[0]
        dm1[i+1] -= dm2[i]/d0[i] * dp1[i]
        d0[i+2] -= dm2[i]/d0[i] * dp2[i]
        # dp1[2] -= dm2[0]/d0[0] * 0
        # dp2[2] -= dm2[0]/d0[0] * 0
    L1.append(dm1[n-2]/d0[n-2])
    d0[n-2+1] -= dm1[n-2]/d0[n-2] * dp1[n-2]
    # dp1[n-2+1] -= dm1[n-2]/do[n-2] * dp2[n-2]

    # dm1[n-2+1] -= dm2[n-2]/d0[n-2] * dp1[n-2]
    # d0[n-2+2] -= dm2[n-2]/d0[n-2] * dp2[n-2]
    # d0, dp1, and dp2 make up U
    # L1 and L2 make up L

    Ld = b
    d = [b[0], b[1] - L1[0]*b[0]]
    for i in np.arange(2, n):
        d.append(b[i] - L2[i-2]*d[i-2] - L1[i-1]*d[i-1])

    Ux = d
    x = np.zeros(n)
    x[-1] = d[-1] / d0[-1]
    x[-2] = ((d[-2] - dp1[-1] * x[-1])/d0[-2])
    for i in np.arange(3, n+1):
        x[-i] = ((d[-i] - dp1[1-i] * x[1-i] - dp2[2-i] * x[2-i])/d0[-i])
    return x

def pent_SOR(A, b, omega, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        x_new = np.zeros(n)
        sum2 = A[0, 1] * x[1] + A[0, 2] * x[2]
        x_new[0] = (1 - omega) * x[0] + omega * (b[0] - sum2) / A[0, 0]
        sum1 = A[1, 0] * x_new[0]
        sum2 = A[1, 2] * x[2] + A[1, 3] * x[3]
        x_new[1] = (1 - omega) * x[1] + omega * (b[1] - sum1 - sum2) / A[1, 1]
        for i in np.arange(2, n-2):
            sum1 = A[i, i-2] * x_new[i-2] + A[i, i-1] * x_new[i-1]
            sum2 = A[i, i+1] * x[i+1] + A[i, i+2] * x[i+2]
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        sum1 = A[-2, -2-2] * x_new[-2-2] + A[-2, -2-1] * x_new[-2-1]
        sum2 = A[-2, -2+1] * x[-2+1]
        x_new[-2] = (1 - omega) * x[-2] + omega * (b[-2] - sum1 - sum2) / A[-2, -2]
        sum1 = A[-1, -1-2] * x_new[-1-2] + A[-1, -1-1] * x_new[-1-1]
        x_new[-1] = (1 - omega) * x[-1] + omega * (b[-1] - sum1) / A[-1, -1]

        err = np.linalg.norm(x_new - x)
        x = x_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')
    return x, iter_count

if part2:
    ## part 2

    n = 5000 # TODO this should be 5000
    A_matrix = make_A(n)
    b = np.random.rand(n)

    start = time.time()
    x = solve_pent_LU(A_matrix, b)
    end = time.time()
    resid = np.linalg.norm(A_matrix @ x - b)
    print(f'Pentdiagonal LU took {end-start} seconds to complete. {resid=}')

    start = time.time()
    x, _ = pent_SOR(A_matrix, b, omeg, 1e-6, max_iter)
    end = time.time()
    resid = np.linalg.norm(A_matrix @ x - b)
    print(f'Pentdiagonal SOR took {end-start} seconds to complete. {resid=}')

    start = time.time()
    x = np.linalg.solve(A_matrix, b)
    end = time.time()
    resid = np.linalg.norm(A_matrix @ x - b)
    print(f'Numpy solver took {end-start} seconds to complete. {resid=}')

    start = time.time()
    x = np.linalg.inv(A_matrix) * b
    end = time.time()
    resid = np.linalg.norm(A_matrix @ x - b)
    print(f'Numpy inverter took {end-start} seconds to complete. {resid=}')

    A_sparse = scipy.sparse.diags_array([-1, 16, -30, 16, -1], offsets=[-2, -1, 0, 1, 2], shape=(n, n), dtype='float64').tocsr()/12
    start = time.time()
    x = scipy.sparse.linalg.spsolve(A_sparse, b)
    end = time.time()
    resid = np.linalg.norm(A_matrix @ x - b)
    print(f'Scipy sparse solver took {end-start} seconds to complete. {resid=}')

if testing:
    A = scipy.sparse.diags_array([-1, 1, -2, 1, -1], offsets=[-2, -1, 0, 1, 2], shape=(5, 5), dtype='float64').toarray()
    print(A)
    b = np.random.rand(5)
    print(b)
    # n = A.shape[0]
    # A = A.astype(float).copy()
    # b = b.astype(float).copy()
    #
    # # LU Decomposition
    # U = A.copy()
    # L = np.eye(n)
    # for i in range(n - 1):
    #     for j in range(i + 1, n):
    #         L[j, i] = U[j, i] / U[i, i]
    #         U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]
    #
    # print("standard LU decomp")
    # print("L")
    # print(L)
    # print("U")
    # print(U)
    # # Forward substitution (Ld = Pb)
    # d = np.zeros(n)
    # for i in range(n):
    #     sum_val = np.dot(L[i, :i], d[:i]) if i > 0 else 0.0
    #     d[i] = b[i] - sum_val
    #
    # # Backward substitution (Ux = d)
    # x = np.zeros(n)
    # for i in range(n - 1, -1, -1):
    #     sum_val = np.dot(U[i, i:], x[i:])
    #     x[i] = (d[i] - sum_val) / U[i, i]
    # print(f'{d=}')
    # print(f'{x=}')
    #
    # dm2 = np.ones(n-2) * A[0,2]
    # dm1 = np.ones(n-1) * A[0,1]
    # d0 = np.ones(n) * A[0,0]
    # dp1 = np.ones(n-1) * A[0,1]
    # dp2 = np.ones(n-2) * A[0,2]
    # L1 = []
    # L2 = []
    #
    # for i in range(n-2):
    #     L1.append(dm1[i]/d0[i])
    #     # dm1[0] -= dm1[0]/d0[0] * d0[0]
    #     d0[i+1] -= dm1[i]/d0[i] * dp1[i]
    #     dp1[i+1] -= dm1[i]/d0[i] * dp2[i]
    #     # dp2[1] -= dm1[0]/d0[0] * 0
    #
    #     L2.append(dm2[i]/d0[i])
    #     # dm2[0] -= dm2[0]/d0[0] * d0[0]
    #     dm1[i+1] -= dm2[i]/d0[i] * dp1[i]
    #     d0[i+2] -= dm2[i]/d0[i] * dp2[i]
    #     # dp1[2] -= dm2[0]/d0[0] * 0
    #     # dp2[2] -= dm2[0]/d0[0] * 0
    # L1.append(dm1[n-2]/d0[n-2])
    # d0[n-2+1] -= dm1[n-2]/d0[n-2] * dp1[n-2]
    # # dp1[n-2+1] -= dm1[n-2]/do[n-2] * dp2[n-2]
    #
    # # dm1[n-2+1] -= dm2[n-2]/d0[n-2] * dp1[n-2]
    # # d0[n-2+2] -= dm2[n-2]/d0[n-2] * dp2[n-2]
    # # d0, dp1, and dp2 make up U
    # # L1 and L2 make up L
    # print(A)
    # print(b)
    # print("efficient LU decomp")
    # print("L")
    # print(L1)
    # print(L2)
    # print("U")
    # print(d0)
    # print(dp1)
    # print(dp2)
    #
    # Ld = b
    # d = [b[0], b[1] - L1[0]*b[0]]
    # for i in np.arange(2, n):
    #     d.append(b[i] - L2[i-2]*d[i-2] - L1[i-1]*d[i-1])
    #
    # Ux = d
    # x = np.zeros(n)
    # x[-1] = d[-1] / d0[-1]
    # x[-2] = ((d[-2] - dp1[-1] * x[-1])/d0[-2])
    # for i in np.arange(3, n+1):
    #     x[-i] = ((d[-i] - dp1[1-i] * x[1-i] - dp2[2-i] * x[2-i])/d0[-i])
    # print(f'{d=}')
    # print(f'{x=}')

    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    err = 1.0
    omega = omeg

    while iter_count < max_iter and err > tol:
        x_new = np.zeros(n)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i]) if i > 0 else 0.0
            sum2 = np.dot(A[i, i+1:], x[i+1:]) if i < n - 1 else 0.0

            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]

        err = np.linalg.norm(x_new - x)
        x = x_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')
    print(f'{x=}\n{iter_count=}')

    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    err = 1.0

    while iter_count < max_iter and err > tol:
        x_new = np.zeros(n)
        sum2 = A[0, 1] * x[1] + A[0, 2] * x[2]
        x_new[0] = (1 - omega) * x[0] + omega * (b[0] - sum2) / A[0, 0]
        sum1 = A[1, 0] * x_new[0]
        sum2 = A[1, 2] * x[2] + A[1, 3] * x[3]
        x_new[1] = (1 - omega) * x[1] + omega * (b[1] - sum1 - sum2) / A[1, 1]
        for i in np.arange(2, n-2):
            sum1 = A[i, i-2] * x_new[i-2] + A[i, i-1] * x_new[i-1]
            sum2 = A[i, i+1] * x[i+1] + A[i, i+2] * x[i+2]
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        sum1 = A[-2, -2-2] * x_new[-2-2] + A[-2, -2-1] * x_new[-2-1]
        sum2 = A[-2, -2+1] * x[-2+1]
        x_new[-2] = (1 - omega) * x[-2] + omega * (b[-2] - sum1 - sum2) / A[-2, -2]
        sum1 = A[-1, -1-2] * x_new[-1-2] + A[-1, -1-1] * x_new[-1-1]
        x_new[-1] = (1 - omega) * x[-1] + omega * (b[-1] - sum1) / A[-1, -1]

        err = np.linalg.norm(x_new - x)
        x = x_new
        iter_count += 1

    if iter_count == max_iter and tol < err:
        warnings.warn('Convergence not reached at maximum iteration count')
    print(f'{x=}\n{iter_count=}')


plt.show()