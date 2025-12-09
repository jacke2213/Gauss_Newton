import numpy as np
from typing import Callable, Tuple
import func_gn

from grad import grad_c, jacobian_c


tol = 1e-10
plotout = True
printout = True

t, y = func_gn.get_data_json("data1.json")
#t, y = func_gn.get_data_json("data2.json")

def armijo(f, x, d):
    eps = 0.2
    alpha = 2
    lamd = 1
    eval = 3

    F_prim0 = np.dot(grad_c(f,x),d)
    F_0 = f(x)

    

    while f(x + alpha * lamd * d) < F_0 + eps * (alpha * lamd) * F_prim0:
        lamd *= alpha
        eval += 1

    while f(x + lamd * d) > F_0 + eps * lamd * F_prim0:
        lamd /= alpha
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
    
    print("Initial gradient:", np.linalg.norm(grad_c(f, x0)))

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

        try:
            d_k = np.linalg.solve(J_2, -J_res)
        except np.linalg.LinAlgError:
            d_k = np.linalg.lstsq(J_2, -J_res, rcond=None)[0]


        evals, x, lamd = armijo(f,x,d_k)
        N_eval += evals

        if printout:
            print(f"{k:3d}   {np.max(np.abs(r(x))):10.4e}   {normg:10.4e}   {evals:3d}   {N_eval:4d}   {lamd:7.4f}")


    if plotout:
        pass

    return x, N_eval, k, normg



# X är 2dimensionell för phi1? och 4-dim för phi2?

x0 = np.array([1,2,3,4])
result = gauss_newton(func_gn.phi1, t, y, x0, tol, printout, plotout)

print(f'\n')
print(result)









