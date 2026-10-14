import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
import math
import scipy.sparse as sps
import scipy.optimize as spo
import scipy.special as spp
import time

w = 0.05
h = 0.25
bottom_spaces = 10
edge_spacing = w/(bottom_spaces)
x_centers = np.linspace(0, w, int(w/edge_spacing) + 1)[:-1] + edge_spacing/2
y_centers = np.linspace(0, h, int(h/edge_spacing) + 1)[:-1] + edge_spacing/2
xspace, yspace = np.meshgrid(x_centers, y_centers)
matrix_T = np.zeros_like(xspace)
vector_T = matrix_T.flatten()
v0 = 0.03
k = 0.606
k_rht = 25
k_top = 50
k_lft = 25
k_bot = 2
k_mat_rht = np.full_like(xspace, k)
k_mat_rht[:,  -1] = k_rht*edge_spacing
k_mat_top = np.full_like(xspace, k)
k_mat_top[-1,  :] = k_top*edge_spacing
k_mat_lft = np.full_like(xspace, k)
k_mat_lft[:,  0] = k_lft*edge_spacing
k_mat_bot = np.full_like(xspace, k)
k_mat_bot[0, :] = k_bot*edge_spacing
implicit_derate = 5
rho = 1000
cp = 4186
T_amb = 25
lower_limit = 1e-20

# print(xspace)
# print(yspace)

def get_vx_vy(x, y, arg_v0):
    vx =  arg_v0 * np.sin(np.pi * x / w) * np.cos(np.pi * y / h)
    vy = -arg_v0 * np.cos(np.pi * x / w) * np.sin(np.pi * y / h) * h / w
    return vx, vy

def get_connectivity(matrix):
    rht = sps.eye_array(matrix.size, k=1).tolil()
    rht[matrix.shape[1]-1:-1:matrix.shape[1],:] = 0
    bot = sps.eye_array(matrix.size, k=-matrix.shape[1])
    lft = sps.eye_array(matrix.size, k=-1).tolil()
    lft[0:-1:matrix.shape[1],:] = 0
    top = sps.eye_array(matrix.size, k=+matrix.shape[1])
    return rht, top, lft, bot

rht, top, lft, bot = get_connectivity(matrix_T)

# test_flat = np.arange(12)
# test = test_flat.reshape((3, 4))
# rht, top, lft, bot = get_connectivity(test)
# print(test)
# print(rht @ test_flat)
# print(top @ test_flat)
# print(lft @ test_flat)
# print(bot @ test_flat)
# print(rht.toarray())
# print(top.toarray())
# print(lft.toarray())
# print(bot.toarray())

# ((vx * T) + (vx+i * T+i))/2 + etc
# vx * T + vy * T + (vx+i * T+i)/2


# print(rht * edge_spacing/edge_spacing + top * edge_spacing/edge_spacing - lft * edge_spacing/edge_spacing - bot * edge_spacing/edge_spacing)
# TODO it mostly seems to work except for the velocities thing. probably an issue because i didn't actually read the equation, try reading the equations.
# advection
# (sps.diags_array(0.5 * (edge_spacing * (-get_vx_vy(xspace-edge_spacing/2, yspace)[0].flatten() + get_vx_vy(xspace+edge_spacing/2, yspace)[0].flatten()) + edge_spacing * (get_vx_vy(xspace, yspace+edge_spacing/2)[1].flatten() + -get_vx_vy(xspace, yspace-edge_spacing/2)[1].flatten()))) + 0.5 * edge_spacing * (lft * -get_vx_vy(xspace-edge_spacing/2, yspace)[0].flatten()[:,None] + rht * get_vx_vy(xspace+edge_spacing/2, yspace)[0].flatten()[:,None]) + 0.5 * edge_spacing * (top * get_vx_vy(xspace, yspace+edge_spacing/2)[1].flatten()[:,None] + bot * -get_vx_vy(xspace, yspace-edge_spacing/2)[1].flatten()[:,None]))

def get_A(arg_v0):
    adv = rho * cp * (sps.diags_array(0.5 * (edge_spacing * (-get_vx_vy(xspace-edge_spacing/2, yspace, arg_v0)[0].flatten() + get_vx_vy(xspace+edge_spacing/2, yspace, arg_v0)[0].flatten()) + edge_spacing * (get_vx_vy(xspace, yspace+edge_spacing/2, arg_v0)[1].flatten() + -get_vx_vy(xspace, yspace-edge_spacing/2, arg_v0)[1].flatten())))\
    + 0.5 * edge_spacing * (lft * -get_vx_vy(xspace-edge_spacing/2, yspace, arg_v0)[0].flatten()[:,None] + rht * get_vx_vy(xspace+edge_spacing/2, yspace, arg_v0)[0].flatten()[:,None]) \
    + 0.5 * edge_spacing * (top * get_vx_vy(xspace, yspace+edge_spacing/2, arg_v0)[1].flatten()[:,None] + bot * -get_vx_vy(xspace, yspace-edge_spacing/2, arg_v0)[1].flatten()[:,None]))
    diff = ((k*rht*edge_spacing/edge_spacing - sps.diags_array(k_mat_rht.flatten())*edge_spacing/edge_spacing) \
    + (k*top*edge_spacing/edge_spacing - sps.diags_array(k_mat_top.flatten())*edge_spacing/edge_spacing) \
    + (k*lft*edge_spacing/edge_spacing - sps.diags_array(k_mat_lft.flatten())*edge_spacing/edge_spacing) \
    + (k*bot*edge_spacing/edge_spacing - sps.diags_array(k_mat_bot.flatten())*edge_spacing/edge_spacing))
    return (adv + diff) / rho / cp / edge_spacing / edge_spacing
matrix_A = get_A(v0)
# matrix_A = (rho * cp  * (sps.diags_array(0.5 * (edge_spacing * (-get_vx_vy(xspace-edge_spacing/2, yspace)[0].flatten() + get_vx_vy(xspace+edge_spacing/2, yspace)[0].flatten()) + edge_spacing * (get_vx_vy(xspace, yspace+edge_spacing/2)[1].flatten() + -get_vx_vy(xspace, yspace-edge_spacing/2)[1].flatten())))\
#     + 0.5 * edge_spacing * (lft * -get_vx_vy(xspace-edge_spacing/2, yspace)[0].flatten()[:,None] + rht * get_vx_vy(xspace+edge_spacing/2, yspace)[0].flatten()[:,None]) \
#     + 0.5 * edge_spacing * (top * get_vx_vy(xspace, yspace+edge_spacing/2)[1].flatten()[:,None] + bot * -get_vx_vy(xspace, yspace-edge_spacing/2)[1].flatten()[:,None])) \
#     + ((k*rht*edge_spacing/edge_spacing - sps.diags_array(k_mat_rht.flatten())*edge_spacing/edge_spacing) \
#     + (k*top*edge_spacing/edge_spacing - sps.diags_array(k_mat_top.flatten())*edge_spacing/edge_spacing) \
#     + (k*lft*edge_spacing/edge_spacing - sps.diags_array(k_mat_lft.flatten())*edge_spacing/edge_spacing) \
#     + (k*bot*edge_spacing/edge_spacing - sps.diags_array(k_mat_bot.flatten())*edge_spacing/edge_spacing))) \
#     / rho / cp / edge_spacing / edge_spacing

vector_F = np.zeros_like(vector_T)
vector_F[matrix_T.shape[1]-1::matrix_T.shape[1]]    += (k_rht * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[-matrix_T.shape[1]:]                       += (k_top * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[0:-1:matrix_T.shape[1]]                    += (k_lft * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[0:matrix_T.shape[1]]                       += (k_bot * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb

# print(matrix_A)
# print(vector_F)


# print(matrix_T)
# print(np.reshape(matrix_A @ vector_T + vector_F, matrix_T.shape))

def step_RK1(vector, A, F, t_step):
    dTdt = A @ vector + F
    return vector + dTdt * t_step

def step_RK2(vector, A, F, t_step):
    k1 = A @ vector + F
    k2 = A @ (vector + k1 * t_step) + F
    dTdt = (k1 + k2)/ 2
    return vector + dTdt * t_step

def step_RK4(vector, A, F, t_step):
    k1 = A @ vector + F
    k2 = A @ (vector + k1 * t_step / 2) + F
    k3 = A @ (vector + k2 * t_step / 2) + F
    k4 = A @ (vector + k3 * t_step) + F
    dTdt = (k1 + 2*k2 + 2*k3 + k4) / 6
    return vector + dTdt * t_step

def step_implicit(vector, A, F, t_step):
    return sps.linalg.spsolve((sps.eye_array(len(vector)) - A*t_step), (vector + t_step*F))

def stability_RK(eig, order):
    return np.abs(np.sum(np.power(eig, np.arange(order+1)) / spp.factorial(np.arange(order+1))))

def stability_limit_RK(eigs, order):
    eig = np.max(eigs)
    if np.abs(eig) <= 1e-16: return np.nan # assume that this is probably zero
    most_stable = spo.minimize_scalar(lambda x : stability_RK(eig * x, order), bounds=(lower_limit, 1e+2), method='bounded')
    if stability_RK(eig * most_stable.x, order) > 1:
        return np.nan, most_stable.x, np.nan
    try:
        limit_lower = spo.brentq(lambda x : stability_RK(eig * x, order) - (1 - 1e-6), lower_limit, most_stable.x) # 1e-6 otherwise the flat (very slightly unstable) portion "looks" stable to the solver
    except ValueError:
        limit_lower = 0
    try:
        limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, most_stable.x + 1e+2)
    except ValueError:
        limit_upper = np.nan
    return limit_lower, most_stable.x, limit_upper

def run_sim(func, order, name, plottimes, derate = 1):
    vector_T = matrix_T.flatten()
    try:
        values = sps.linalg.eigs(matrix_A, which="LM", k=2, return_eigenvectors=False)
    except:
        values = np.linalg.eigvals(matrix_A.toarray())
    min_step, opt_step, max_step = stability_limit_RK(values, order)
    t_step = plottimes[1] / np.ceil(plottimes[1] / opt_step / derate)
    history = [matrix_T.copy()]
    plotsteps = np.round(plottimes / t_step)
    for i in np.arange(np.max(plotsteps)):
        vector_T = func(vector_T, matrix_A, vector_F, t_step)
        if i in plotsteps:
            history.append(np.reshape(vector_T, matrix_T.shape))
    data = np.asarray(history)
    norm = plt.Normalize(vmin=data.min(), vmax = data.max())
    fig, ax = plt.subplots(1, len(plottimes), sharey=True)
    mesh = None
    for index in np.arange(len(plottimes)):
        mesh = ax[index].pcolor(xspace, yspace, history[index])
        ax[index].set_title(str(plottimes[index]) + " s")
    fig.colorbar(mesh, ax=ax, label="Temperature (C)")
    print(f"finished running {name} with {t_step=}")

def timetest(func, name, order, endtime, derate = 1):
    timestart = time.time()
    vector_T = matrix_T.flatten()
    try:
        values = sps.linalg.eigs(matrix_A, which="LM", k=2, return_eigenvectors=False)
    except:
        values = np.linalg.eigvals(matrix_A.toarray())
    min_step, opt_step, max_step = stability_limit_RK(values, order)
    t_step = endtime / np.ceil(endtime / opt_step / derate)
    for i in np.arange(int(endtime / t_step)):
        vector_T = func(vector_T, matrix_A, vector_F, t_step)
    timeend = time.time()
    print(f"Time to complete 60s using {name}: {timeend - timestart}, timestep = {t_step}")
    print(f"Simulation speed = {endtime / (timeend - timestart)}")



plottimes = np.asarray([0, 60, 600, 1800, 3600])
run_sim(step_RK4, 4, "RK4", plottimes)
run_sim(step_implicit, 4, "implicit", plottimes, derate = implicit_derate)

try:
    values = sps.linalg.eigs(matrix_A, which="LM", k=2, return_eigenvectors=False)
except:
    values = np.linalg.eigvals(matrix_A.toarray())
print("The minimum, most stable, and maximum stable timestep for RK1 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 1))
print("The minimum, most stable, and maximum stable timestep for RK2 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 2))
print("The minimum, most stable, and maximum stable timestep for RK4 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 4))

timetest(step_RK1, "RK1", 1, 60)
timetest(step_RK2, "RK2", 2, 60)
timetest(step_RK4, "RK4", 4, 60)
timetest(step_implicit, "implicit", 4, 60, derate = implicit_derate)

def get_max_tstep_v(arg_v0):
    new_A = get_A(arg_v0)
    try:
        values = sps.linalg.eigs(new_A, which="LM", k=2, return_eigenvectors=False)
    except:
        values = np.linalg.eigvals(new_A.toarray())
    print(f"The maximum time step for {arg_v0=} is {stability_limit_RK(values, 4)[1]}")
get_max_tstep_v(0)
get_max_tstep_v(0.003)
get_max_tstep_v(0.03)
get_max_tstep_v(0.3)
get_max_tstep_v(3)
get_max_tstep_v(30)
get_max_tstep_v(300)



plt.show()

# mmk so flatten our t matrix (presumably along with xspace and yspace)
# then for each of the four sides, find average T, average vx, and average vy
# really we just want one matrix out of it
# but I guess we can do four matrices, left right top bottom averages
# ugh that's a lot of recalc though? but idk a more efficient way to do it
# ok so lrtb T average, lr vx average, tb vy average
# first term is lT*lvx + rT*rvx + tT*tvy + bT*bvy
# then second term would be uh
# need to find slope -- could use central difference? but I don't think
# that's what he was teaching us in class tho... ugh
# we could make a whole new proper central difference matrices and then average
# those as well... but tbh kinda lazy don't really wanna do that lol
# ok so once we have lrtb pderivs then second term is
# lT*ldx + rT*rdx + tT*tdy + bT*bdy
# and that leaves us with the left hand side
# so then we just find the next T matrix that satisfies right hand side?

