# AI use statement: Google Gemini 3.1 Pro was used to convert the original MATLAB script to Python with the following prompt: "Can you convert the attached MATLAB script into a Python script? Ideally, it should be possible to do a line-to-line comparison of the two scripts wherever possible."

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import scipy.optimize as spo
import scipy.special as spp


# %% User Inputs

# Material and BC Coefficients
rho = 1000
c_p = 4186
k_al = 0.606
beta = 0.00021
g = -9.81
mu = 0.89
h_side = 25
h_bottom = 2
h_top = 50
s = 0
lower_limit = 1e-20

D = np.array([[k_al, 0], [0, k_al]])

T_amb = 25

# %% Mesh Definition
w_tot = 0.05
h_tot = 0.25
# Define mesh dimensions
nx_nodes = 11  # Number of nodes in the x-direction
ny_nodes = 51  # Number of nodes in the y-direction
dx = w_tot / (nx_nodes - 1)
dy = h_tot / (ny_nodes - 1)
n_nodes = nx_nodes * ny_nodes
ny_el = ny_nodes - 1
nx_el = nx_nodes - 1
n_els = nx_el * ny_el
n_el_nodes = 4

COORDS = np.zeros((n_nodes, 2))

for i in range(n_nodes):
    y_row = i // nx_nodes
    COORDS[i, 0] = (i - y_row * nx_nodes) * dx
    COORDS[i, 1] = y_row * dy

# Initialize connectivity matrix
ELEMS = np.zeros((n_els, 4), dtype=int)

# Loop to fill in the connectivity matrix
elementCount = 0
for j in range(ny_el):
    for i in range(nx_el):
        n1 = j * nx_nodes + i
        n2 = n1 + 1
        n3 = n2 + nx_nodes
        n4 = n1 + nx_nodes

        ELEMS[elementCount, :] = [n1, n2, n3, n4]
        elementCount += 1

# Display the mesh details
print('Node Coordinates:')
print(COORDS)

print('Element Connectivity:')
print(ELEMS)

fig, ax = plt.subplots()
for el in range(n_els):
    el_coords = COORDS[ELEMS[el, :], :]
    poly = Polygon(el_coords, closed=True, facecolor='none', edgecolor='red')
    ax.add_patch(poly)

for i in range(n_nodes):
    ax.plot(COORDS[i, 0], COORDS[i, 1], 'ob', markersize=5)
    ax.text(COORDS[i, 0] + 0.00025, COORDS[i, 1] + 0.00025, str(i + 1)) # +1 to match MATLAB node numbering display

rect = Rectangle((0, 0), w_tot, h_tot, edgecolor='b', facecolor='none', linewidth=2)
ax.add_patch(rect)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal')
# plt.show()

# %% PART 1 - You will need to update this code to incorporate transient effects.

# Generate shape function and derivative matrices and Integrate
K_el = np.zeros((n_el_nodes, n_el_nodes, n_els))
K_k_el = np.zeros((n_el_nodes, n_el_nodes, n_els))
K_q_el = np.zeros((n_el_nodes, n_el_nodes, n_els))
F_el_Omeg = np.zeros((n_el_nodes, n_els))
F_el_Gamma = np.zeros((n_el_nodes, n_els))

C_el = np.zeros((n_el_nodes, n_el_nodes, n_els))

# Loop Through Elements to Find Source/Sink Force Vector and Stiffness Matrix
# Using Isoparametric Elements so initialize gauss pts
WTS = [1, 1]  # Weights are 1 for 4pt integration (2pts in each dir)
GP = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

for el in range(n_els):
    # Get nodal Coords
    coords = COORDS[ELEMS[el, :], :]

    # Loop Through Gauss Points and Integrate
    for i in range(2):
        for j in range(2):
            xi = GP[i]
            eta = GP[j]

            # Shape Functions
            N = 0.25 * np.array([[(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]])

            # Bprime matrix (G*N in book)
            Bprime = 0.25 * np.array([[eta - 1, 1 - eta, 1 + eta, -eta - 1],
                                      [xi - 1, -xi - 1, 1 + xi, 1 - xi]])

            # Jacobian
            J_el = Bprime @ coords

            det_J = np.linalg.det(J_el)
            if det_J < 0:
                raise ValueError('Negative element jacobian')

            J_el_inv = np.linalg.inv(J_el)

            # Generate B Matrix
            B = J_el_inv @ Bprime

            # Integration Point of our Gauss Point
            # Stiffness
            K_k_el[:, :, el] += WTS[i] * WTS[j] * B.T @ D @ B * det_J

            # Source/Sink element Force
            F_el_Omeg[:, el] += (WTS[i] * WTS[j] * N.T * s * det_J).flatten()

            C_el[:, :, el] += WTS[i] * WTS[j] * N.T * rho * c_p @ N * det_J

# %% Calculate flux BC on Right Side
for el in range(9, n_els, 10):  # 10:10:n_els in MATLAB -> 9, 19, 29... in Python
    coords = COORDS[ELEMS[el, :], :]

    for i in range(2):
        xi = 1
        eta = GP[i]

        N = 0.25 * np.array([[(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]])
        Bprime = 0.25 * np.array([[eta - 1, 1 - eta, 1 + eta, -eta - 1],
                                  [xi - 1, -xi - 1, 1 + xi, 1 - xi]])

        J_el = Bprime @ coords

        det_J = (coords[2, 1] - coords[1, 1]) / 2  # Note this is different because it's a line integral
        if det_J < 0:
            raise ValueError('Negative element jacobian')

        # Integration
        # MATLAB used WTS(j) here which was uninitialized in the scope, assuming WTS[i] for line integral
        K_q_el[:, :, el] += WTS[i] * WTS[i] * (N.T @ N) * h_side * det_J
        F_el_Gamma[:, el] += (WTS[i] * h_side * N.T * T_amb * det_J).flatten()

# %% Calculate flux BC on Left Side
for el in range(0, n_els, 10):  # 1:10:n_els in MATLAB -> 0, 10, 20... in Python
    coords = COORDS[ELEMS[el, :], :]

    for i in range(2):
        xi = -1
        eta = GP[i]

        N = 0.25 * np.array([[(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]])
        Bprime = 0.25 * np.array([[eta - 1, 1 - eta, 1 + eta, -eta - 1],
                                  [xi - 1, -xi - 1, 1 + xi, 1 - xi]])

        J_el = Bprime @ coords

        det_J = (coords[3, 1] - coords[0, 1]) / 2
        if det_J < 0:
            raise ValueError('Negative element jacobian')

        K_q_el[:, :, el] += WTS[i] * WTS[i] * (N.T @ N) * h_side * det_J
        F_el_Gamma[:, el] += (WTS[i] * h_side * N.T * T_amb * det_J).flatten()

# %% Calculate flux BC on Top
for el in range(n_els - 10, n_els):  # n_els-9:n_els in MATLAB -> last 10 elements
    coords = COORDS[ELEMS[el, :], :]

    for i in range(2):
        xi = GP[i]
        eta = 1

        N = 0.25 * np.array([[(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]])
        Bprime = 0.25 * np.array([[eta - 1, 1 - eta, 1 + eta, -eta - 1],
                                  [xi - 1, -xi - 1, 1 + xi, 1 - xi]])

        J_el = Bprime @ coords

        det_J = (coords[2, 0] - coords[3, 0]) / 2
        if det_J < 0:
            raise ValueError('Negative element jacobian')

        K_q_el[:, :, el] += WTS[i] * WTS[i] * (N.T @ N) * h_top * det_J
        F_el_Gamma[:, el] += (WTS[i] * h_top * N.T * T_amb * det_J).flatten()

# %% Calculate flux BC on Bottom
for el in range(10):  # 1:10 in MATLAB -> first 10 elements
    coords = COORDS[ELEMS[el, :], :]

    for i in range(2):
        xi = GP[i]
        eta = -1

        N = 0.25 * np.array([[(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]])
        Bprime = 0.25 * np.array([[eta - 1, 1 - eta, 1 + eta, -eta - 1],
                                  [xi - 1, -xi - 1, 1 + xi, 1 - xi]])

        J_el = Bprime @ coords

        det_J = (coords[1, 0] - coords[0, 0]) / 2

        if det_J < 0:
            raise ValueError('Negative element jacobian')

        K_q_el[:, :, el] += WTS[i] * WTS[i] * (N.T @ N) * h_bottom * det_J
        F_el_Gamma[:, el] += (WTS[i] * h_bottom * N.T * T_amb * det_J).flatten()

K_el = K_k_el + K_q_el

# %% Now Assemble
K = np.zeros((n_nodes, n_nodes))
C = np.zeros((n_nodes, n_nodes))
F = np.zeros(n_nodes)

for el in range(n_els):
    for node in range(n_el_nodes):
        I = ELEMS[el, node]
        F[I] += F_el_Omeg[node, el] + F_el_Gamma[node, el]
        for node2 in range(n_el_nodes):
            J = ELEMS[el, node2]
            K[I, J] += K_el[node, node2, el]
            C[I, J] += C_el[node, node2, el]

d = np.full((n_nodes), 0)
def step_RK1(d_arg, C_inv_arg, K_arg, F_arg, dt_arg):
    d_dot = C_inv_arg @ (K_arg @ d_arg - F_arg)
    d_out = d_arg - d_dot * dt
    return d_out
dt = 1
t = 0
C_inv = np.linalg.inv(C)

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

eigs, _ = np.linalg.eig(-C_inv @ K)
print(stability_limit_RK(eigs, 1))
print(stability_limit_RK(eigs, 2))
print(stability_limit_RK(eigs, 4))
print(np.max(eigs))
history_points = [0, 60, 600, 1800, 3600]
history = [d.copy()]

while t < 3600:
    d = step_RK1(d, C_inv, K, F, dt)
    t += dt
    if t in history_points: history.append(d.copy())

def weird_pcolor(solution):
    # %% Post Process
    fig2, ax2 = plt.subplots()

    for el in range(n_els):
        el_coords = COORDS[ELEMS[el, :], :]
        d_el = solution[ELEMS[el, :]]

        # Use fill for pcolormesh approximation. Here we map the nodes using polygon fill
        dmax = 0
        if (dmax := np.max(np.abs(d))) == 0:
            dmax = 1e-12
        # dmax = T_amb
        poly = Polygon(el_coords, closed=True, facecolor=plt.cm.viridis(np.mean(d_el) / dmax), edgecolor='k')
        ax2.add_patch(poly)

    ax2.set_xlim([0, w_tot])
    ax2.set_ylim([0, h_tot])
    ax2.set_aspect('equal')

    # Create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=np.min(d), vmax=np.max(d)))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2)
    return fig2, ax2

for i in np.arange(len(history)):
    figl, axl = weird_pcolor(history[i])
    axl.set_title(f"t = {history_points[i]} secconds")

    # breakpoint()



# Solve (For part 2- you will need solve this using the Jacobian, Gauss-Seidel, and SOR Method).
# d = np.linalg.solve(K, -F)



plt.show()