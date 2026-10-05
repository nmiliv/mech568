import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
import math
import scipy.sparse as sps

w = 0.05
h = 0.25
spacing = 0.05/3
xpoints = np.linspace(0, w, int(w/spacing))
ypoints = np.linspace(0, h, int(h/spacing))
xspace, yspace = np.meshgrid(xpoints, ypoints)
matrix_T = np.zeros_like(xspace)
v0 = 0.03
v0 = 0
k = 0.606
k_lft = np.full_like(matrix_T, k)
k_lft[:,0] = 25
k_rht = np.full_like(matrix_T, k)
k_rht[:,-1] = 25
k_top = np.full_like(matrix_T, k)
k_top[0,:] = 50
k_bot = np.full_like(matrix_T, k)
k_bot[-1,:] = 2

rho = 1000
cp = 4186
vx =  v0 * np.sin(np.pi * xspace / w) * np.cos(np.pi * yspace / h)
vy = -v0 * np.cos(np.pi * xspace / w) * np.sin(np.pi * yspace / h)
T_amb = 25

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
    limit_lower = spo.brentq(lambda x : stability_RK(eig * x, order) - (1 - 1e-6), lower_limit, most_stable.x) # 1e-6 otherwise the flat (very slightly unstable) portion "looks" stable to the solver
    limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, most_stable.x + 1e+2)
    return limit_lower, most_stable.x, limit_upper


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

def get_T_aves(matrix):
    # alrighty so left average T... T + slide to the right divide by two
    lft_ave_T = (matrix + np.hstack([np.full((matrix.shape[0], 1), T_amb), matrix[:,:-1]])) / 2
    rht_ave_T = (matrix + np.hstack([matrix[:,+1:], np.full((matrix.shape[0], 1), T_amb)])) / 2
    top_ave_T = (matrix + np.vstack([np.full((1, matrix.shape[1]), T_amb), matrix[:-1,:]])) / 2
    bot_ave_T = (matrix + np.vstack([matrix[+1:,:], np.full((1, matrix.shape[1]), T_amb)])) / 2
    print("-------------T_ave-----------------")
    print(lft_ave_T)
    print(rht_ave_T)
    print(top_ave_T)
    print(bot_ave_T)
    return lft_ave_T, top_ave_T, rht_ave_T, bot_ave_T

def get_v_aves(v_x, v_y):
    lft_ave_vx = (v_x + np.hstack([np.full((v_x.shape[0], 1), 0), v_x[:,:-1]])) / 2 # TODO are these zeros on the boundary accurate? or should they just copy the closest value?
    rht_ave_vx = (v_x + np.hstack([v_x[:,+1:], np.full((v_x.shape[0], 1), 0)])) / 2
    top_ave_vy = (v_y + np.vstack([np.full((1, v_y.shape[1]), 0), v_y[:-1,:]])) / 2
    bot_ave_vy = (v_y + np.vstack([v_y[+1:,:], np.full((1, v_y.shape[1]), 0)])) / 2
    print("-------------v_ave-----------------")
    print(lft_ave_vx)
    print(rht_ave_vx)
    print(top_ave_vy)
    print(bot_ave_vy)
    return lft_ave_vx, top_ave_vy, rht_ave_vx, bot_ave_vy

def get_pdrs(matrix, space): # TODO ok got it: use the xpos and ypos matrices to figure out spacing. when rolling the matrices, just use spacing to fill in the missinc row/col. This should overcome the whole "what is positive" issue.
    lft_pdr_T = (+matrix - np.hstack([np.full((matrix.shape[0], 1), T_amb), matrix[:,:-1]])) / space
    rht_pdr_T = (-matrix + np.hstack([matrix[:,+1:], np.full((matrix.shape[0], 1), T_amb)])) / space
    top_pdr_T = (-matrix + np.vstack([np.full((1, matrix.shape[1]), T_amb), matrix[:-1,:]])) / space
    bot_pdr_T = (+matrix - np.vstack([matrix[+1:,:], np.full((1, matrix.shape[1]), T_amb)])) / space
    print("-------------pdr-----------------")
    print(lft_pdr_T)
    print(rht_pdr_T)
    print(top_pdr_T)
    print(bot_pdr_T)
    return lft_pdr_T, top_pdr_T, rht_pdr_T, bot_pdr_T

def get_rhs(matrix, v_x, v_y):
    lft_ave_T, top_ave_T, rht_ave_T, bot_ave_T = get_T_aves(matrix)
    lft_ave_vx, top_ave_vy, rht_ave_vx, bot_ave_vy = get_v_aves(v_x, v_y)
    lft_pdr_T, top_pdr_T, rht_pdr_T, bot_pdr_T = get_pdrs(matrix, spacing)
    print("-------------term 1------------")
    print(spacing*(rht_ave_T*rht_ave_vx - lft_ave_T*lft_ave_vx - top_ave_T*top_ave_vy + bot_ave_T*bot_ave_vy))
    print("-------------term 2------------")
    print(spacing*(k_rht*rht_pdr_T - k_lft*lft_pdr_T - k_top*top_pdr_T + k_bot*bot_pdr_T) / rho / cp)
    print("-------------term 2 rht------------")
    print(spacing*(k_rht*rht_pdr_T) / rho / cp)
    print("-------------term 2 lft------------")
    print(spacing*(- k_lft*lft_pdr_T) / rho / cp)
    print("-------------term 2 top ------------")
    print(spacing*(- k_top*top_pdr_T) / rho / cp)
    print("-------------term 2 bot------------")
    print(spacing*(k_bot*bot_pdr_T) / rho / cp)
    right_hand_side = spacing*(lft_ave_T*lft_ave_vx - rht_ave_T*rht_ave_vx - top_ave_T*top_ave_vy + bot_ave_T*bot_ave_vy) + spacing*(k_lft*lft_pdr_T - k_rht*rht_pdr_T - k_top*top_pdr_T + k_bot*bot_pdr_T) / rho / cp
    print("-------------rhs-----------------")
    print(right_hand_side)
    return right_hand_side

def step_RK1(matrix, v_x, v_y, t_step):
    next_step = get_rhs(matrix, v_x, v_y) / spacing ** 2 * t_step + matrix
    print("-------------next step-----------------")
    print(next_step)
    return next_step

plots = 3
history = [matrix_T.copy()]
t_step = 0.1
for i in np.arange(0, plots*1-1):
    matrix_T = step_RK1(matrix_T, vx, vy, t_step)
    history.append(matrix_T)



# nextstep = right_hand_side / spacing**2 * 4 * spacing * tstep + matrix_T
# TODO turn this into a loop
# TODO tstep
# TODO turn this into an A matrix
# TODO boundary conditions
# TODO find stability
# TODO check that the averages are done correctly
# TODO currently FE/RK1, need to adjust to RK4
# ideas on how to fully matrixify
# should be possible to do all of our lrtb averages as matrix ops on vectors of space
# but will probably need to build these matrices by hand? idk should be possible to do
# row/column ops on the identity to get connectivity
# ugh idk just get the basic stuff working for now and then play around with that later
# and convert to sparse matrices lol

# print(matrix_T)
# print(lft_ave_T)
# print(rht_ave_T)
# print(top_ave_T)
# print(bot_ave_T)


fig, ax = plt.subplots(1, 2)
ax[0].pcolor(xspace, yspace, vx)
ax[1].pcolor(xspace, yspace, vy)

fig, ax = plt.subplots(1, plots)
for index in np.arange(plots):
    ax[index].pcolor(xspace, yspace, history[index])
print("-------------history-----------------")
print(history)
plt.show()
breakpoint()