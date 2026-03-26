#%% imports
import vedo as vd
import numpy as np
from vedo.pyplot import plot
from vedo import Latex
import time

# set backend
vd.settings.default_backend = 'vtk'

#%% Global variables
msg = vd.Text2D(pos='bottom-left', font="VictorMono")  # status text
X0_guess = None                   # initial guess (x,y)
step_size = 0.1                   # GD line-search initial step
n_steps = 5                       # not used here
tol = 1e-6                        # convergence
epsilon = 1e-6                    # LM regularizer
max_iter = 1000                   # iteration cap

#%% Helper functions

def gradient_fd(func, X, h=1e-3):
    gx = (func([X[0]+h, X[1]]) - func([X[0]-h, X[1]])) / (2*h)
    gy = (func([X[0], X[1]+h]) - func([X[0], X[1]-h])) / (2*h)
    return np.array([gx, gy])

def Hessian_fd(func, X, h=1e-3):
    x, y = X
    gxx = (func([x+h,y]) - 2*func([x,y]) + func([x-h,y]))/h**2
    gyy = (func([x,y+h]) - 2*func([x,y]) + func([x,y-h]))/h**2
    gxy = (func([x+h,y+h]) - func([x+h,y-h]) - func([x-h,y+h]) + func([x-h,y-h]))/(4*h**2)
    return np.array([[gxx, gxy],[gxy, gyy]])

# Analytical derivatives

def gradient_an(X):
    x, y = X
    gx = y * np.cos(2*x*y) * np.cos(3*y)
    gy = x * np.cos(2*x*y) * np.cos(3*y) - 1.5 * np.sin(2*x*y) * np.sin(3*y)
    return np.array([gx, gy])

def Hessian_an(X):
    x, y = X
    f_xx = -2*y**2 * np.sin(2*x*y) * np.cos(3*y)
    f_yy = -2*x**2 * np.sin(2*x*y) * np.cos(3*y) - 6*x*np.cos(2*x*y)*np.sin(3*y) - 4.5*np.sin(2*x*y)*np.cos(3*y)
    f_xy = np.cos(2*x*y)*np.cos(3*y) - 2*x*y*np.sin(2*x*y)*np.cos(3*y) - 3*y*np.cos(2*x*y)*np.sin(3*y)
    return np.array([[f_xx, f_xy],[f_xy, f_yy]])

def line_search(func, X, d, alpha0=1.0, c=1e-4, rho=0.5):
    fx = func(X)
    grad = gradient_fd(func, X)
    alpha = alpha0
    while func(X + alpha*d) > fx + c * alpha * np.dot(grad, d):
        alpha *= rho
        if alpha < 1e-12:
            break
    return alpha

#%% Objective
def objective(X): return np.sin(2*X[0]*X[1]) * np.cos(3*X[1]) / 2 + 0.5

#%% Numeric vs Analytic timing & accuracy test
Xtest = [0.5, -0.3]
print("Timing 10000 calls (average of 5 runs):")
for name, fn in [('FD grad', lambda: gradient_fd(objective, Xtest)),
                 ('AN grad', lambda: gradient_an(Xtest)),
                 ('FD Hess', lambda: Hessian_fd(objective, Xtest)),
                 ('AN Hess', lambda: Hessian_an(Xtest))]:
    times = []
    for _ in range(5):
        t0 = time.time()
        for _ in range(10000): fn()
        times.append(time.time() - t0)
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"{name}: {avg_time:.4f}s ± {std_time:.4f}s")

print("\nFD grad accuracy:")
an_grad = gradient_an(Xtest)
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    fdg = gradient_fd(objective, Xtest, h=h)
    err = np.linalg.norm(fdg - an_grad)
    print(f"h={h:.0e}, grad err={err:.2e}")

print("\nFD Hessian accuracy:")
an_hess = Hessian_an(Xtest)
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    fdh = Hessian_fd(objective, Xtest, h=h)
    err = np.linalg.norm(fdh - an_hess, ord='fro')
    print(f"h={h:.0e}, Hessian err={err:.2e}")

#%% Task 4 Routines

def run_gd(X0):
    X = np.array(X0)
    path = [np.append(X, objective(X))]
    for k in range(max_iter):
        g = gradient_fd(objective, X)
        if np.linalg.norm(g) < tol:
            return path, k
        alpha = line_search(objective, X, -g)
        X = X - alpha * g
        path.append(np.append(X, objective(X)))
    return path, max_iter

def run_newton(X0):
    """
    Newton's method using analytic gradient and Hessian.
    """
    X = np.array(X0)
    path = [np.append(X, objective(X))]
    for k in range(max_iter):
        g = gradient_an(X)                        # analytic gradient
        if np.linalg.norm(g) < tol:
            return path, k
        H = Hessian_an(X)                        # analytic Hessian
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return path, k
        alpha = line_search(objective, X, d)
        X = X + alpha * d
        path.append(np.append(X, objective(X)))
    return path, max_iter

def run_modnewton1(X0):
    """
    Modified Newton: fallback to gradient if Hessian not positive definite, using analytic derivatives.
    """
    X = np.array(X0)
    path = [np.append(X, objective(X))]
    for k in range(max_iter):
        g = gradient_an(X)                        # analytic gradient
        if np.linalg.norm(g) < tol:
            return path, k
        H = Hessian_an(X)                         # analytic Hessian
        eigs = np.linalg.eigvalsh(H)
        d = -g if np.any(eigs <= 0) else -np.linalg.solve(H, g)
        alpha = line_search(objective, X, d)
        X = X + alpha * d
        path.append(np.append(X, objective(X)))
    return path, max_iter

def run_modnewton2(X0):
    """
    Levenberg–Marquardt modification: shift Hessian to be positive definite, using analytic derivatives.
    """
    X = np.array(X0)
    path = [np.append(X, objective(X))]
    for k in range(max_iter):
        g = gradient_an(X)                        # analytic gradient
        if np.linalg.norm(g) < tol:
            return path, k
        H = Hessian_an(X)                         # analytic Hessian
        eigs, _ = np.linalg.eigh(H)
        delta = max(0, -np.min(eigs) + epsilon)
        M = H + delta*np.eye(2)
        try:
            d = -np.linalg.solve(M, g)
        except np.linalg.LinAlgError:
            # fallback to steepest descent if still singular
            d = -g
        alpha = line_search(objective, X, d)
        X = X + alpha * d
        path.append(np.append(X, objective(X)))
    return path, max_iter

#%% Callbacks & UI for Task 4 Evaluation
plt = vd.Plotter(bg2='lightblue')
# 3D surface
surf = plot(lambda x,y: objective([x,y]), c='terrain')
surf[0].name='Surface'; surf[1].name='Isolines'
# flat floor
proj = surf.clone()
proj[0].lighting('off'); proj[0].vertices[:,2]=0
proj[1].vertices[:,2]=0
# placeholders
eg = plot([0,1],[0,1e-3],title='Energy vs Iter',xtitle='Iter',ytitle='E')
eg2 = eg.clone2d(); eg2.name='EnergyGraph'
dp = plot([0,1],[-1,1],title='Dot Prod',xtitle='Iter',ytitle='∇f·d')
dp2 = dp.clone2d(); dp2.name='DotGraph'

# callbacks

def OnMouseMove(evt):
    if evt.object and evt.picked3d is not None:
        x,y,_ = evt.picked3d
        msg.text(f"X: ({x:.2f},{y:.2f}), E={objective([x,y]):.2f}")
        plt.render()

def OnLeftClick(evt):
    global X0_guess
    if evt.object and evt.picked3d is not None:
        x,y,_ = evt.picked3d
        X0_guess = (x,y)
        msg.text(f"Guess: ({x:.2f},{y:.2f})")
        plt.remove('InitMarker')
        mk = vd.Sphere(pos=(x,y,0),r=0.03,c='red')
        mk.name = 'InitMarker'
        plt.add(mk)
        plt.render()

def OnEvaluate(*args):
    if X0_guess is None:
        msg.text("Set initial guess first.")
        plt.render()
        return
    results = {}
    for name, fn in [('GD', run_gd), ('Newton', run_newton), ('ModNewton1', run_modnewton1), ('ModNewton2', run_modnewton2)]:
        t0 = time.time()
        path, iters = fn(X0_guess)
        results[name] = (iters, time.time()-t0, path)
    report = "Iters & Time:\n"
    for name, (it, tm, _) in results.items():
        report += f"{name}: {it} iters in {tm:.3f}s\n"
    plt.remove('EvalText')
    et = vd.Text2D(report, pos=(0.02,0.8), font="VictorMono")
    et.name = 'EvalText'
    plt.add(et)
    # plot each energy curve separately
    plt.remove('EnergyGraph')
    maxlen = max(len(p) for _,_,p in results.values())
    xs = np.arange(maxlen)
    colors = {'GD':'orange','Newton':'blue','ModNewton1':'purple','ModNewton2':'green'}
    for name,(_,_,p) in results.items():
        vals = [pt[2] for pt in p]
        vals += [vals[-1]]*(maxlen-len(vals))
        g = plot(xs, vals, c=colors[name], legend=name)
        grp = g.clone2d()
        grp.name = 'EnergyGraph'
        plt.add(grp)
    # dot-product for ModNewton2
    plt.remove('DotGraph')
    path2 = results['ModNewton2'][2]
    dps = []
    for pt in path2[1:]:
        grad = gradient_fd(objective, pt[:2])
        H = Hessian_fd(objective, pt[:2])
        d = -np.linalg.solve(H + epsilon*np.eye(2), grad)
        dps.append(np.dot(grad, d))
    if dps:
        xs2 = np.arange(1, len(dps)+1)
        dp_plot = plot(xs2, dps, c='red', legend='∇f·d')
        dpgrp = dp_plot.clone2d()
        dpgrp.name = 'DotGraph'
        plt.add(dpgrp)
    plt.render()

# register callbacks & widgets
plt.add_callback('mouse move', OnMouseMove)
plt.add_callback('mouse left click', OnLeftClick)
plt.add_button(OnEvaluate, states=['Evaluate'], pos=(0.75, 0.90))  # moved slightly left so full button shows
plt.add_slider(lambda w,e: globals().update({'step_size':w.value}) or msg.text(f"Step size={w.value:.2f}") or plt.render(),
               0.001, 1, step_size, title='Step Size', pos='bottom-left')
plt.add_slider(lambda w,e: globals().update({'n_steps':int(w.value)}) or msg.text(f"Steps={int(w.value)}") or plt.render(),
               1, 50, n_steps, title='Num Steps', pos='top-left')
plt.add_slider(lambda w,e: surf[0].alpha(w.value) or surf[1].alpha(w.value) or plt.render(),
               0, 1, 1, title='Alpha', pos='bottom-right')

# show
plt.show([surf, proj, eg2, dp2], msg, __doc__, viewup='z')
plt.close()