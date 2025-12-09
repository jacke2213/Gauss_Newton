import numpy as np
from typing import Callable, Tuple
import func_gn

from grad import grad_c, jacobian_c


tol = 1.e-4
plotout = True
printout = True

t, y = func_gn.get_data_json("data1.json")
#t, y = func_gn.get_data_json("data2.json")

def armijo(f, x, d):
    eps = 0.2
    alpha = 2
    lamd = 1
    eval = 3
    
    F = lambda lam: f(x + d*lam)
    F_prim0 = np.dot(grad_c(f,x),d)
    F_0 = F(0)

    while F(alpha*lamd) < F_0 + eps * F_prim0*alpha*lamd:
        lamd = alpha * lamd
        eval += 1
    while F(lamd) > F_0 + eps * F_prim0 * lamd:
        lamd = lamd / alpha
        eval += 1

    return eval, x + d * lamd, lamd


def gauss_newton(phi : Callable[[np.ndarray, np.ndarray], np.ndarray], 
    t : np.ndarray,
    y : np.ndarray, 
    x0 : np.ndarray, 
    tol : float, 
    printout : bool,
    plotout : bool) -> Tuple[np.ndarray, int, int, float]:
    
    N_iter = 0
    N_eval = 0
    max_iter = 100
    
    r = lambda x : phi(x, t) - y
    f = lambda x : np.sum(r(x)**2)

    x = x0.copy()

    if printout:
            print("iter     x              max(abs(r))   norm(grad)   ls   fun evals   lamb")

    for k in range(max_iter):
        N_iter += 1
        grad_fk = grad_c(f,x)
        normg = np.linalg.norm(grad_fk)
        if normg < tol:
            break

        J = jacobian_c(r, x) 
        J_t = np.transpose(J)
        J_2 = np.matmul(J_t, J) 
        
        res = r(x)
        J_res = np.matmul(J_t, res)

        d_k = np.linalg.solve(J_2, -J_res)

        evals, x, lamd = armijo(f,x,d_k)
        N_eval += evals

        """if printout:
            max_abs_r = float(np.max(np.abs(res)))
            # första raden
            print(
                f"{k:3d}   {x[0]:.4f}        {max_abs_r:10.4f}   {normg:10.4f}   "
                f"{evals:2d}   {N_eval:4d}        {lamd:6.4f}"
            )
            for j in range(1, len(x)):
                print(f"      {x[j]:.4f}")"""


    if plotout:
        pass

    return x, N_eval, k, normg

x0 = np.array([1,2,3,4])
result = gauss_newton(func_gn.phi1, t, y, x0, tol, printout, plotout)

print(result)









