import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
import math
import scipy.sparse as sps

w = 0.05
h = 0.25
bottom_spaces = 3
edge_spacing = w/(bottom_spaces)
x_centers = np.linspace(0, w, int(w/edge_spacing) + 1)[:-1] + edge_spacing/2
y_centers = np.linspace(0, h, int(h/edge_spacing) + 1)[:-1] + edge_spacing/2
xspace, yspace = np.meshgrid(x_centers, y_centers)
matrix_T = np.zeros_like(xspace)
v0 = 0.03
v0 = 0
k = 0.606
k_rht = 25
k_top = 50
k_lft = 25
k_bot = 2
# k_matrix = np.full_like(xspace, k)
# k_matrix[:,  0] = 25
# k_matrix[:, -1] = 25
# k_matrix[0,  :] = 50
# k_matrix[-1, :] = 2

print(xspace)
print(yspace)

rho = 1000
cp = 4186
def get_vx_vy(x, y):
    vx =  v0 * np.sin(np.pi * x / w) * np.cos(np.pi * y / h)
    vy = -v0 * np.cos(np.pi * x / w) * np.sin(np.pi * y / h)
    return vx, vy
vx, vy = get_vx_vy(xspace, yspace)
T_amb = 25

def get_connectivity(matrix):
    rht = sps.eye_array(matrix.size, k=1).tolil()
    rht[matrix.shape[1]-1:-1:matrix.shape[1],:] = 0
    top = sps.eye_array(matrix.size, k=-matrix.shape[1])
    lft = sps.eye_array(matrix.size, k=-1).tolil()
    lft[0:-1:matrix.shape[1],:] = 0
    bot = sps.eye_array(matrix.size, k=+matrix.shape[1])
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

vector_T = matrix_T.flatten()
vector_vx = vx.flatten()
vector_vy = vy.flatten()


# print(rht * edge_spacing/edge_spacing + top * edge_spacing/edge_spacing - lft * edge_spacing/edge_spacing - bot * edge_spacing/edge_spacing)

matrix_A = (rho * cp  * (np.eye(matrix_T.size) @ (vector_vx*edge_spacing/2 + vector_vy*edge_spacing/2 - vector_vx*edge_spacing/2 - vector_vy*edge_spacing/2) \
    + rht @ (vector_vx*edge_spacing/2) + top @ (vector_vy*edge_spacing/2) - lft @ (vector_vx*edge_spacing/2) - bot @ (vector_vy*edge_spacing/2)) \
        + k * (rht * edge_spacing/edge_spacing + top * edge_spacing/edge_spacing - lft * edge_spacing/edge_spacing - bot * edge_spacing/edge_spacing)) / rho / cp / edge_spacing / edge_spacing
    # + k * ("""np.eye(matrix_T.size) @ (-edge_spacing/edge_spacing - edge_spacing/edge_spacing + edge_spacing/edge_spacing + edge_spacing/edge_spacing)""" + rht * edge_spacing/edge_spacing + top * edge_spacing/edge_spacing - lft * edge_spacing/edge_spacing - bot * edge_spacing/edge_spacing)) / rho / cp / edge_spacing / edge_spacing

vector_F = np.zeros_like(vector_T)
vector_F[matrix_T.shape[1]-1::matrix_T.shape[1]]    += (rho * cp * vx[:,-1]*edge_spacing/2 + k_rht * edge_spacing/edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[-matrix_T.shape[1]:]                       += (rho * cp * vx[-1,:]*edge_spacing/2 + k_top * edge_spacing/edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[0:-1:matrix_T.shape[1]]                    += (rho * cp * vx[:, 0]*edge_spacing/2 + k_lft * edge_spacing/edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[0:matrix_T.shape[1]]                       += (rho * cp * vy[0, :]*edge_spacing/2 + k_bot * edge_spacing/edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb


# print(matrix_A)
# print(vector_F)

# print(matrix_T)
# print(np.reshape(matrix_A @ vector_T + vector_F, matrix_T.shape))

def step_RK1(vector, A, F, t_step):
    dTdt = A @ vector + F
    return vector + dTdt * t_step

def step_simple(matrix, t_step, v_x, v_y):
    first_term = np.zeros_like(matrix)
    second_term = np.zeros_like(matrix)
    for yin in np.arange(0, matrix.shape[0]):
        for xin in np.arange(0, matrix.shape[1]-1):
            first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin, xin+1]*matrix[yin, xin+1])/2 * edge_spacing
            second_term[yin, xin] += k * (matrix[yin, xin+1] - matrix[yin, xin]) / edge_spacing * edge_spacing
    for yin in np.arange(0, matrix.shape[0]):
        xin = matrix.shape[1]-1
        first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin, xin]*T_amb)/2 * edge_spacing
        second_term[yin, xin] += k_rht * (T_amb - matrix[yin, xin]) / edge_spacing * edge_spacing

    for yin in np.arange(0, matrix.shape[0]-1):
        for xin in np.arange(0, matrix.shape[1]):
            first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin+1, xin]*matrix[yin+1, xin])/2 * -edge_spacing
            second_term[yin, xin] += k * (matrix[yin+1, xin] - matrix[yin, xin]) / edge_spacing * -edge_spacing
    for xin in np.arange(0, matrix.shape[1]):
        yin = matrix.shape[0]-1
        first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin, xin]*T_amb)/2 * -edge_spacing
        second_term[yin, xin] += k_top * (T_amb - matrix[yin, xin]) / edge_spacing * -edge_spacing

    for yin in np.arange(0, matrix.shape[0]):
        for xin in np.arange(1, matrix.shape[1]):
            first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin, xin-1]*matrix[yin, xin-1])/2 * -edge_spacing
            second_term[yin, xin] += k * (matrix[yin, xin-1] - matrix[yin, xin]) / -edge_spacing * -edge_spacing
    for yin in np.arange(0, matrix.shape[0]):
        xin = 0
        first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin, xin]*T_amb)/2 * -edge_spacing
        second_term[yin, xin] += k_lft * (T_amb - matrix[yin, xin]) / -edge_spacing * -edge_spacing

    for yin in np.arange(1, matrix.shape[0]):
        for xin in np.arange(0, matrix.shape[1]):
            first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin-1, xin]*matrix[yin-1, xin])/2 * edge_spacing
            second_term[yin, xin] += k * (matrix[yin-1, xin] - matrix[yin, xin]) / -edge_spacing * edge_spacing
    for xin in np.arange(0, matrix.shape[1]):
        yin = 0
        first_term += rho * cp * (v_x[yin, xin]*matrix[yin, xin] + v_x[yin, xin]*T_amb)/2 * edge_spacing
        second_term[yin, xin] += k_bot * (T_amb - matrix[yin, xin]) / -edge_spacing * edge_spacing
    dTdt = first_term + second_term / rho / cp / edge_spacing / edge_spacing
    return matrix + dTdt * t_step

plots = 3
history = [matrix_T.copy()]
t_step = 0.1
for i in np.arange(0, plots*5-1):
    vector_T = step_RK1(vector_T, matrix_A, vector_F, t_step)
    history.append(np.reshape(vector_T, matrix_T.shape))
print(history)

plots = 3
history_2 = [matrix_T.copy()]
t_step = 0.1
for i in np.arange(0, plots*5-1):
    matrix_T = step_simple(matrix_T, t_step, vx, vy)
    history_2.append(matrix_T)
print(history_2)

fig, ax = plt.subplots(1, plots)
for index in np.arange(plots):
    ax[index].pcolor(xspace, yspace, history[index])

fig, ax = plt.subplots(1, plots)
for index in np.arange(plots):
    ax[index].pcolor(xspace, yspace, history_2[index])

plt.show()

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
#
#
# fig, ax = plt.subplots(1, 2)
# ax[0].pcolor(xspace, yspace, vx)
# ax[1].pcolor(xspace, yspace, vy)













