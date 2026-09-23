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

lower_limit = 1e-20

def stability_limit_RK(eigs, order):
    eig = np.max(eigs)
    if np.abs(eig) <= 1e-16: return np.nan # assume that this is probably zero
    most_stable = spo.minimize_scalar(lambda x : stability_RK(eig * x, order), bounds=(lower_limit, 1e+2), method='bounded')
    if stability_RK(eig * most_stable.x, order) > 1:
        return np.nan, most_stable.x, np.nan
    limit_lower = spo.brentq(lambda x : stability_RK(eig * x, order) - (1 + 1e-6), lower_limit, most_stable.x)
    limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, 1e+2)
    return limit_lower, most_stable.x, limit_upper
    # eigs = [np.max(eigs)]
    # limits = []
    # for eig in eigs:
    #     if np.abs(eig) <= 1e-16: continue # assume that this is probably zero
    #     most_stable = spo.minimize_scalar(lambda x : stability_RK(eig * x, order), bounds=(lower_limit, 1e+2), method='bounded')
    #     if stability_RK(eig * most_stable.x, order) > 1:
    #         limits.append(np.nan)
    #         continue
    #     limit_lower = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, lower_limit, most_stable.x)
    #     limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, 1e+2)
    #     limit = limit_upper
    #     limits.append(limit)
    # # print(limits)
    # return np.max(limits)


def build_b(xspace, bounds, stencil_size):
    a_sparse = sps.lil_array((xpoints, xpoints))
    for loc in np.arange(xpoints):
        template = np.arange(stencil_size)
        if loc - (stencil_size - 1)/2 < 0:
            template = xspace[0:stencil_size] - xspace[loc]
            a_sparse[loc,0:stencil_size] = template_to_stencil(template, 2)
        elif loc + (stencil_size - 1)/2 >= xpoints:
            template = xspace[-stencil_size:]
            a_sparse[loc,-stencil_size:] = template_to_stencil(template, 2)
        else:
            template = xspace[int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] - xspace[loc]
            a_sparse[loc,int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] =  template_to_stencil(template, 2)
    a_sparse = sps.coo_array(a_sparse[np.where(np.isnan(bounds))])
    a_sparse = a_sparse[:, np.where(np.isnan(bounds))]
    a_sparse = a_sparse[:,0,:]
    b_sparse = sps.block_array([[sps.coo_array((xpoints-2, xpoints-2)), sps.eye_array(xpoints-2)], [c**2*a_sparse, sps.coo_array((xpoints-2, xpoints-2))]])
    return b_sparse # TODO return f matrix as well -- should be as simpleas just identifying what we threw out in truncation

xspace = np.linspace(0, L, xpoints)
bounds = np.full_like(xspace, np.nan)
bounds[0] = 0
bounds[-1] = p_init
b_sparse = build_b(xpsace, bounds, 3)
values = sps.linalg.eigs(b_sparse, which="LM", k=2, return_eigenvectors=False)

print("The minimum, most stable, and maximum stable timestep for RK1 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 1))
print("The minimum, most stable, and maximum stable timestep for RK2 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 2))
print("The minimum, most stable, and maximum stable timestep for RK4 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 4))

xpoints_space = np.logspace(int(np.log10(L/1e+1/0.3048)), int(np.log10(L/1e-4/0.3048)), num=50).astype(np.int_) + 1
max_dts = []
for xpoints in xpoints_space:
    if len(max_dts) > 0 and np.isnan(max_dts[-1]):
        max_dts.append(np.nan)
        continue
    xspace = np.linspace(0, L, xpoints)
    bounds = np.full_like(xspace, np.nan)
    bounds[0] = 0
    bounds[-1] = p_init
    b_sparse = build_b(xspace, bounds, 3)
    print(f'trying npoints = {xpoints}')
    try:
        values = sps.linalg.eigs(b_sparse, which="LM", k=2, return_eigenvectors=False, maxiter=100)
    except:
        max_dts.append(np.nan)
        print(f"could not converge npoints = {xpoints}, aborting larger matrices")
    else:
        max_dts.append(stability_limit_RK(values, 4))

max_dts = np.asarray(max_dts)

# print(max_dts)
# print(np.where(np.isfinite(max_dts)))
# print(xpoints_space[np.where(np.isfinite(max_dts))])
m, b = np.polyfit(np.log(xpoints_space[np.where(np.isfinite(max_dts))]), np.log(max_dts[np.where(np.isfinite(max_dts))]), 1)
print(m)
print(b)

fig, ax = plt.subplots(2)
ax[0].loglog(xpoints_space, max_dts)
ax[0].loglog(xpoints_space, math.exp(b) * np.power(xpoints_space, m))
ax[1].semilogx(xpoints_space, max_dts - math.exp(b) * np.power(xpoints_space, m))
print(xpoints_space)
print(max_dts)
plt.show()

def step_rk4(B, y, dt): # asked AI to check whether this implementation was correct
    k1 = B.multiply(y)
    k2 = B.multiply(y + dt/2 * k1)
    k3 = B.multiply(y + dt/2 * k2)
    k4 = B.multiply(y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

b_sparse = build_b(int(1e+2 + 1), 3)
values = sps.linalg.eigs(b_sparse, which="LM", k=2, return_eigenvectors=False)
tspace = np.arange(0, 0.05, stability_limit_RK(values, 4))
dt = stability_limit_RK(values, 4)[1]
time = 0
init = np.full(101, p_init)
bounds = np.full_like(init, np.nan)
bounds[0] = 0
bounds[-1] = p_init
results = []
# eugh ok so how do we enforce the boundary conditions?
#
while time < 0.05:



#
# # find stencil
# # then populate A matrix
# # apply boundaries (replace rows)
# # build B matrix through block ops
# # find eig, check if each eig is stable (for known dt) or find dt needed for stability
#
# def template_to_stencil(template, order):
#     stencil = np.linalg.inv(np.transpose(np.vander(template)))[:,-order-1]*math.factorial(order)
#     return stencil
# # print(template_to_stencil([-1,0,1], 1))
#
# xspace = np.linspace(0, L, xpoints)
# bounds = np.full_like(xspace, np.nan)
# bounds[0] = 0
# bounds[-1] = p_init
#
# stencil_size = 3
# # for each point
# # find stencil_size nearest points
# # use that to calc template
# # use template and order to find stencil
# # stencil can be centered if loc - (half size - 1) >= 0 or loc + (half size - 1) < xpoints
# # a_matrix = np.identity(xpoints)
#
# a_sparse = sps.lil_array((xpoints, xpoints))
#
# for loc in np.arange(xpoints):
#     if np.isfinite(bounds[loc]):
#         # TODO aw hell what do we do at the boundary again? do we really want to drop that row? or i think we zero it out but ugh the 2nd deriv on the boundary isn't really defined in any real way?? well the 2nd deriv in time is fixed (zero) so I guess by definition the second deriv in space on those rows will also be zero?
#         # oh wait nvm we just delete those rows/cols, so set to whatever here then use bounds to properly delete later?
#         # a_matrix[loc,loc] = 0
#         continue
#     template = np.arange(stencil_size)
#     if loc - (stencil_size - 1)/2 < 0:
#         template = xspace[0:stencil_size] - xspace[loc]
#         # a_matrix[loc,0:stencil_size] = template_to_stencil(template, 2)
#         a_sparse[loc,0:stencil_size] = template_to_stencil(template, 2)
#     elif loc + (stencil_size - 1)/2 >= xpoints:
#         template = xspace[-stencil_size:]
#         # a_matrix[loc,-stencil_size:] = template_to_stencil(template, 2)
#         a_sparse[loc,-stencil_size:] = template_to_stencil(template, 2)
#     else:
#         template = xspace[int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] - xspace[loc]
#         # a_matrix[loc,int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] =  template_to_stencil(template, 2)
#         a_sparse[loc,int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] =  template_to_stencil(template, 2)
#     # print(a_sparse)
# # print(np.isnan(bounds))
# # print(np.where(np.isnan(bounds)))
# # a_matrix = a_matrix[np.where(np.isnan(bounds)), np.where(np.isnan(bounds))]
# # a_matrix = a_matrix[np.where(np.isnan(bounds))]
# a_sparse = sps.coo_array(a_sparse[np.where(np.isnan(bounds))])
# # print(a_matrix)
# # a_matrix = a_matrix[:, np.where(np.isnan(bounds))]
# a_sparse = a_sparse[:, np.where(np.isnan(bounds))]
# # a_matrix = a_matrix[:,0,:] # dunno why i have to do this but i end up with double nesting otherwise
# a_sparse = a_sparse[:,0,:]
# # print(a_matrix)
# # print(a_sparse)
# # print(a_matrix - a_sparse)
# # b_matrix = np.block([[np.zeros_like(a_matrix), np.identity(np.shape(a_matrix)[0])], [c**2*a_matrix, np.zeros_like(a_matrix)]])
# b_sparse = sps.block_array([[sps.coo_array((xpoints-2, xpoints-2)), sps.eye_array(xpoints-2)], [c**2*a_sparse, sps.coo_array((xpoints-2, xpoints-2))]])
# # print(b_matrix)
#
# # print(npl.norm(a_matrix - a_sparse.toarray()))
# # print(npl.norm(b_matrix - b_sparse.toarray()))
# print("finding eigs")
# # values_d, vectors_d = np.linalg.eig(b_matrix)
# values, vectors = sps.linalg.eigs(b_sparse, which="LM")
# # print(values)
# # print(vectors)
#
# stabspace = np.logspace(-10, 2, 200)
# stability = []
# for point in stabspace:
#     stability.append(stability_RK(values[0] * point, 8))
# fig, ax = plt.subplots(1)
# ax.loglog(stabspace, stability)
# ax.set_ylim([0.5, 2])
# # breakpoint()
# plt.show()