import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
import math
import scipy.optimize as spo
import scipy.special as spp
import scipy.sparse as sps

c = 1125 * 12 * 25.4 / 1000 # ft/s * in/ft * mm/in * m/mm = m/s, sound speed constant by definition
p_init = 850 / 14.7 * 101325 # psi * atm/psi * Pa/atm = Pa, initial pressure in tube
L = 100 * 12 * 25.4 / 1000 # ft * in/ft * mm/in * m/mm = m, length of tube

def template_to_stencil(template, order):
    return np.linalg.inv(np.transpose(np.vander(template)))[:,-order-1]*math.factorial(order)

def stability_RK(eig, order):
    return np.abs(np.sum(np.power(eig, np.arange(order+1)) / spp.factorial(np.arange(order+1))))

def stability_limit_RK(eigs, order):
    eig = np.max(eigs)
    if np.abs(eig) <= 1e-16: return np.nan # assume that this is probably zero
    most_stable = spo.minimize_scalar(lambda x : stability_RK(eig * x, order), bounds=(lower_limit, 1e+2), method='bounded')
    if stability_RK(eig * most_stable.x, order) > 1:
        return np.nan, most_stable.x, np.nan
    limit_lower = spo.brentq(lambda x : stability_RK(eig * x, order) - (1 - 1e-6), lower_limit, most_stable.x) # 1e-6 otherwise the flat (very slightly unstable) portion "looks" stable to the solver
    limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, 1e+2)
    return limit_lower, most_stable.x, limit_upper

def build_b(xspace, bounds, stencil_size):
    xpoints = len(xspace)
    a_sparse = sps.lil_array((len(xspace), len(xspace)))
    for loc in np.arange(len(xspace)):
        template = np.arange(stencil_size)
        if loc - (stencil_size - 1)/2 < 0:
            template = xspace[0:stencil_size] - xspace[loc]
            a_sparse[loc,0:stencil_size] = template_to_stencil(template, 2)
        elif loc + (stencil_size - 1)/2 >= len(xspace):
            template = xspace[-stencil_size:]
            a_sparse[loc,-stencil_size:] = template_to_stencil(template, 2)
        else:
            template = xspace[int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] - xspace[loc]
            a_sparse[loc,int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] =  template_to_stencil(template, 2)
    a_sparse = sps.coo_array(a_sparse[np.where(np.isnan(bounds))]) # remove rows we aren't calculating
    newbounds = bounds.copy()
    newbounds[np.where(np.isnan(newbounds))] = 0
    f_vector = c**2 * a_sparse @ newbounds
    # print(a_sparse)
    # print(f_vector)
    f_vector[np.where(np.isnan(f_vector))] = 0
    # breakpoint()
    a_sparse = a_sparse[:, np.where(np.isnan(bounds))]
    a_sparse = a_sparse[:,0,:]
    b_sparse = sps.block_array([[sps.coo_array((xpoints-2, xpoints-2)), sps.eye_array(xpoints-2)], [c**2*a_sparse, sps.coo_array((xpoints-2, xpoints-2))]])
    g_vector = np.block([np.zeros_like(f_vector), f_vector])
    return b_sparse, g_vector

def step_rk4(B, y, dt, g): # asked AI to check whether this implementation was correct
    k1 = B @ y + g
    k2 = B @ (y + dt/2 * k1) + g
    k3 = B @ (y + dt/2 * k2) + g
    k4 = B @ (y + dt * k3) + g
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

xpoints = 101
xspace = np.linspace(0, L, xpoints)
bounds = np.full_like(xspace, np.nan)
bounds[0] = 0
bounds[-1] = p_init
b_sparse, g_vector = build_b(xspace, bounds, 3)
values = sps.linalg.eigs(b_sparse, which="LM", k=2, return_eigenvectors=False)
_, dt, _ = stability_limit_RK(values, 4)
time = 0
init = np.full(xpoints, p_init)
init[0] = 0
results = [init]
row = init[1:-1]
row = np.block([row, np.zeros_like(row)])
# row = np.block([row, ])
times = [0]
# print(results)
# print(row)

print(row)
print(b_sparse @ row)

while time < 0.05:
    row = step_rk4(b_sparse, row, dt, g_vector)
    trunc_row = bounds.copy()
    trunc_row[np.where(np.isnan(trunc_row))] = row[:np.sum(np.isnan(trunc_row))]
    results.append(trunc_row)
    time += dt
    times.append(time)


def makeplot(fullsim, xpos, tpos, tstep, L_points, imporexp, plot_density=None, name=None):
    xpos = np.array(xpos)
    if np.size(xpos[0]) == 1:
        xpos, tpos = np.meshgrid(xpos, tpos)
    else:
        _, tpos = np.meshgrid(xpos[0], tpos) # xpos already meshgrid-friendly
    fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
    ax[1].set_zorder(2)
    for index in np.linspace(0, len(tpos)-1-1e-6, 10):
        ax[1].plot(xpos[int(np.floor(index))]/0.3048, fullsim[int(np.floor(index))]/6894.757, label=f"t={tpos[int(np.floor(index)), 0]:.5f}")
    # ax[1].legend(loc="lower right")
    ax[1].legend()
    ax[0].set_title(f"{imporexp}, dt={tstep}, xpoints={L_points}")
    ax[0].set_ylabel("Time, seconds")
    ax[1].set_ylabel("Pressure, PSI")
    ax[1].set_xlabel("Position, ft")
    conts = ax[0].contourf(xpos/0.3048, tpos, fullsim/6894.757)
    cbar = fig.colorbar(conts, ax=[ax[0], ax[1]])
    cbar.ax.set_ylabel("Pressure, PSI")
    if plot_density != None:
        if plot_density == "single":
            histax = ax[1].twinx()
            histax.set_ylabel("Grid density")
            histax.hist(xpos[:-1]/0.3048, bins=15, label="grid density", alpha=0.5)
            histax.set_zorder(1)
        elif plot_density == "overlay":
            ax[0].scatter(xpos/0.3048, tpos, c="red", alpha=0.5, s=1)
    if name != None:
        fig.savefig("plots/" + name + ".png")
        plt.close(fig)
    else:
        plt.show()

makeplot(np.array(results), xspace, times, dt, xpoints, "RK4")