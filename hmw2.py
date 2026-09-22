import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.optimize as spo
import scipy.special as spp

c = 1125 * 12 * 25.4 / 1000 # ft/s * in/ft * mm/in * m/mm = m/s, sound speed constant by definition
p_init = 850 / 14.7 * 101325 # psi * atm/psi * Pa/atm = Pa, initial pressure in tube
L = 100 * 12 * 25.4 / 1000 # ft * in/ft * mm/in * m/mm = m, length of tube

xpoints = 101

# find stencil
# then populate A matrix
# apply boundaries (replace rows)
# build B matrix through block ops
# find eig, check if each eig is stable (for known dt) or find dt needed for stability

def template_to_stencil(template, order):
    stencil = np.linalg.inv(np.transpose(np.vander(template)))[:,-order-1]*math.factorial(order)
    return stencil
# print(template_to_stencil([-1,0,1], 1))

xspace = np.linspace(0, L, xpoints)
bounds = np.full_like(xspace, np.nan)
bounds[0] = 0
bounds[-1] = p_init

stencil_size = 3
# for each point
# find stencil_size nearest points
# use that to calc template
# use template and order to find stencil
# stencil can be centered if loc - (half size - 1) >= 0 or loc + (half size - 1) < xpoints
a_matrix = np.identity(xpoints)
for loc in np.arange(xpoints):
    if np.isfinite(bounds[loc]):
        # TODO aw hell what do we do at the boundary again? do we really want to drop that row? or i think we zero it out but ugh the 2nd deriv on the boundary isn't really defined in any real way?? well the 2nd deriv in time is fixed (zero) so I guess by definition the second deriv in space on those rows will also be zero?
        # oh wait nvm we just delete those rows/cols, so set to whatever here then use bounds to properly delete later?
        a_matrix[loc,loc] = 0
        continue
    template = np.arange(stencil_size)
    if loc - (stencil_size - 1)/2 < 0:
        template = xspace[0:stencil_size] - xspace[loc]
        a_matrix[loc,0:stencil_size] = template_to_stencil(template, 2)
    elif loc + (stencil_size - 1)/2 >= xpoints:
        template = xspace[-stencil_size:]
        a_matrix[loc,-stencil_size:] = template_to_stencil(template, 2)
    else:
        template = xspace[int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] - xspace[loc]
        a_matrix[loc,int(loc-(stencil_size-1)/2):int(loc+(stencil_size-1)/2)+1] =  template_to_stencil(template, 2)
# print(np.isnan(bounds))
# print(np.where(np.isnan(bounds)))
# a_matrix = a_matrix[np.where(np.isnan(bounds)), np.where(np.isnan(bounds))]
a_matrix = a_matrix[np.where(np.isnan(bounds))]
# print(a_matrix)
a_matrix = a_matrix[:, np.where(np.isnan(bounds))]
a_matrix = a_matrix[:,0,:] # dunno why i have to do this but i end up with double nesting otherwise
# print(a_matrix)
b_matrix = np.block([[np.zeros_like(a_matrix), np.identity(np.shape(a_matrix)[0])], [c**2*a_matrix, np.zeros_like(a_matrix)]])
# print(b_matrix)
values, vectors = np.linalg.eig(b_matrix)
# print(values)
# print(vectors)

def stability_RK(eig, order):
    return np.abs(np.sum(np.power(eig, np.arange(order+1)) / spp.factorial(np.arange(order+1))))

lower_limit = 1e-20

def stability_limit_RK(eigs, order):
    limits = []
    for eig in eigs:
        if np.abs(eig) <= 1e-16: continue # assume that this is probably zero
        most_stable = spo.minimize_scalar(lambda x : stability_RK(eig * x, order), bounds=(lower_limit, 1e+2), method='bounded')
        if stability_RK(eig * most_stable.x, order) > 1:
            limits.append(np.nan)
            continue
        limit_lower = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, lower_limit, most_stable.x)
        limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, 1e+2)
        limit = limit_upper
        limits.append(limit)
    print(limits)
    return np.max(limits)

print(stability_limit_RK(values, 1))
print(stability_limit_RK(values, 2))
print(stability_limit_RK(values, 4))

stabspace = np.logspace(-10, 2, 200)
stability = []
for point in stabspace:
    stability.append(stability_RK(values[0] * point, 8))
fig, ax = plt.subplots(1)
ax.loglog(stabspace, stability)
ax.set_ylim([0.5, 2])
# breakpoint()
plt.show()