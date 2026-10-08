import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
import math
import scipy.sparse as sps
import scipy.optimize as spo
import scipy.special as spp

w = 0.05
h = 0.25
bottom_spaces = 10
edge_spacing = w/(bottom_spaces)
x_centers = np.linspace(0, w, int(w/edge_spacing) + 1)[:-1] + edge_spacing/2
y_centers = np.linspace(0, h, int(h/edge_spacing) + 1)[:-1] + edge_spacing/2
xspace, yspace = np.meshgrid(x_centers, y_centers)
matrix_T = np.zeros_like(xspace)
v0 = 0.03
# v0 = 0
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
# k_matrix = np.full_like(xspace, k)
# k_matrix[:,  0] = 25
# k_matrix[:, -1] = 25
# k_matrix[0,  :] = 50
# k_matrix[-1, :] = 2

# print(xspace)
# print(yspace)

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

vector_T = matrix_T.flatten()
vector_vx = vx.flatten()
vector_vy = vy.flatten()


# print(rht * edge_spacing/edge_spacing + top * edge_spacing/edge_spacing - lft * edge_spacing/edge_spacing - bot * edge_spacing/edge_spacing)
# TODO it mostly seems to work except for the velocities thing. probably an issue because i didn't actually read the equation, try reading the equations.
matrix_A = (rho * cp  * (sps.diags_array(vector_vx*edge_spacing/2 + vector_vy*edge_spacing/2 + vector_vx*edge_spacing/2 + vector_vy*edge_spacing/2) \
    + rht @ (sps.diags_array(vector_vx)*edge_spacing/2) + top @ (sps.diags_array(vector_vy)*edge_spacing/2) + lft @ (sps.diags_array(vector_vx)*edge_spacing/2) + bot @ (sps.diags_array(vector_vy)*edge_spacing/2)) \
    + ((k*rht*edge_spacing/edge_spacing - sps.diags_array(k_mat_rht.flatten())*edge_spacing/edge_spacing) \
    + (k*top*edge_spacing/edge_spacing - sps.diags_array(k_mat_top.flatten())*edge_spacing/edge_spacing) \
    + (k*lft*edge_spacing/edge_spacing - sps.diags_array(k_mat_lft.flatten())*edge_spacing/edge_spacing) \
    + (k*bot*edge_spacing/edge_spacing - sps.diags_array(k_mat_bot.flatten())*edge_spacing/edge_spacing))) \
    / rho / cp / edge_spacing / edge_spacing

vector_F = np.zeros_like(vector_T)
vector_F[matrix_T.shape[1]-1::matrix_T.shape[1]]    += (rho * cp * vx[:,-1]*edge_spacing/2 + k_rht * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[-matrix_T.shape[1]:]                       += (rho * cp * vx[-1,:]*edge_spacing/2 + k_top * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[0:-1:matrix_T.shape[1]]                    += (rho * cp * vx[:, 0]*edge_spacing/2 + k_lft * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb
vector_F[0:matrix_T.shape[1]]                       += (rho * cp * vy[0, :]*edge_spacing/2 + k_bot * edge_spacing/edge_spacing * edge_spacing) / rho / cp / edge_spacing / edge_spacing * T_amb


# print(matrix_A)
# print(vector_F)


# print(matrix_T)
# print(np.reshape(matrix_A @ vector_T + vector_F, matrix_T.shape))

def step_RK1(vector, A, F, t_step):
    dTdt = A @ vector + F
    return vector + dTdt * t_step

def step_RK4(vector, A, F, t_step):
    k1 = A @ vector + F
    k2 = A @ (vector + k1 * t_step / 2) + F
    k3 = A @ (vector + k2 * t_step / 2) + F
    k4 = A @ (vector + k3 * t_step) + F
    dTdt = (k1 + 2*k2 + 2*k3 + k4) / 6
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
total = plots*50 *4*60
history = [matrix_T.copy()]
t_step = 0.1
for i in np.arange(total):
    vector_T = step_RK1(vector_T, matrix_A, vector_F, t_step)
    history.append(np.reshape(vector_T, matrix_T.shape))
# print(history)
data = np.asarray(history)
norm = plt.Normalize(vmin=data.min(), vmax=data.max())

fig, ax = plt.subplots(1, plots+1)
mesh = None
for index in np.arange(plots+1):
    mesh = ax[index].pcolor(xspace, yspace, history[int(index*total/plots)], norm=norm)
fig.colorbar(mesh, ax=ax, label="Temperature")

plots = 3
total = plots*50 *4*60
history = [matrix_T.copy()]
t_step = 0.1
for i in np.arange(total):
    vector_T = step_RK4(vector_T, matrix_A, vector_F, t_step)
    history.append(np.reshape(vector_T, matrix_T.shape))
# print(history)
data = np.asarray(history)
norm = plt.Normalize(vmin=data.min(), vmax=data.max())

fig, ax = plt.subplots(1, plots+1)
mesh = None
for index in np.arange(plots+1):
    mesh = ax[index].pcolor(xspace, yspace, history[int(index*total/plots)], norm=norm)
fig.colorbar(mesh, ax=ax, label="Temperature")


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
    try:
        limit_upper = spo.brentq(lambda x : stability_RK(eig * x, order) - 1, most_stable.x, most_stable.x + 1e+2)
    except ValueError:
        limit_upper = np.nan
    return limit_lower, most_stable.x, limit_upper

values = sps.linalg.eigs(matrix_A, which="LM", k=2, return_eigenvectors=False)
print("The minimum, most stable, and maximum stable timestep for RK1 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 1))
print("The minimum, most stable, and maximum stable timestep for RK2 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 2))
print("The minimum, most stable, and maximum stable timestep for RK4 in seconds is (nan for always unstable)")
print(stability_limit_RK(values, 4))

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













