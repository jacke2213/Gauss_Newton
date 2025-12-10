import numpy as np
from typing import Callable, Tuple
import func_gn
import matplotlib.pyplot as plt

from grad import grad_c, jacobian_c


tol = 1e-4
plotout = True
printout = True


t, y = func_gn.get_data_json("data2.json")

def armijo(f, x, d, grad):
    eps = 0.2
    alpha = 2
    lamd = 1
    eval = 2
    
    F = lambda lam: f(x + d*lam)
    F_prim0 = np.dot(grad,d)
    F_0 = F(0)

    while F(alpha*lamd) < F_0 + eps * F_prim0*alpha*lamd and eval < 20:
        lamd = alpha * lamd
        eval += 1
    while F(lamd) > F_0 + eps * F_prim0 * lamd and eval < 20:
        lamd = lamd / alpha
        eval += 1

    return eval, lamd


def gauss_newton(phi : Callable[[np.ndarray, np.ndarray], np.ndarray], 
    t : np.ndarray,
    y : np.ndarray, 
    x0 : np.ndarray, 
    tol : float, 
    printout : bool,
    plotout : bool) -> Tuple[np.ndarray, int, int, float]:
    
    N_iter = 0
    N_eval = 0
    max_iter = 200
    
    r = lambda x : phi(x, t) - y
    f = lambda x : np.sum(r(x)**2)

    x = x0.astype(float).copy()

    if printout:
            print("iter     x              max(abs(r))   norm(grad)   ls   fun evals   lamb")
 
    for k in range(max_iter):
        grad_fk = grad_c(f,x)
        normg = np.linalg.norm(grad_fk)
        if normg < tol:
            break

        J = jacobian_c(r, x)                 
        res = r(x)
       
        d_k, *_ = np.linalg.lstsq(J, -res, rcond=None)

        evals, lamd = armijo(f,x,d_k, grad_fk)
        N_eval += evals
        x = x + d_k * lamd
        N_iter += 1

        if printout:
            max_abs_r = float(np.max(np.abs(res)))
            # första raden
            print(
                f"{N_iter:3d}   {x[0]:.4f}        {max_abs_r:10.4f}   {normg:10.4f}   "
                f"{evals:2d}   {N_eval:4d}        {lamd:6.4f}"
            )
            for j in range(1, len(x)):
                print(f"   {x[j]:10.4f}")
 


    if plotout:
        t_plot = np.linspace(t.min(), t.max(), 200)

        y_curve = phi(x, t_plot)

        plt.figure(figsize=(8,5))
        plt.scatter(t,y, color="blue", s=25, label="datapunkter")

        plt.plot(t_plot, y_curve, color="red", linewidth=2, label="Gauss-Newton approximation")

        plt.xlabel("t")
        plt.ylabel("y")
        plt.title("datapunkter och approximerad funktion")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return x, N_eval, N_iter, normg

x0 = np.array([1,2,3,4])
gauss_newton(func_gn.phi2, t, y, x0, tol, printout, plotout)











