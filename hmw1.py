import numpy as np
import math
import matplotlib.pyplot as plt
import time

# AI use: I occasionally used it to refresh my memory of mathematical operation
# names, such as the Vandermonde matrix used in calculating finite difference
# stencils. All code was written by me.

c = 1125 * 12 * 25.4 / 1000 # ft/s * in/ft * mm/in * m/mm = m/s, sound speed constant by definition
p_init = 850 / 14.7 * 101325 # psi * atm/psi * Pa/atm = Pa, initial pressure in tube
L = 100 * 12 * 25.4 / 1000 # ft * in/ft * mm/in * m/mm = m, length of tube

def iterate_explicit(row, lastrow, boundary, xpos, t_now, t_row, t_lastrow):
    if len(row) != len(boundary) or len(row) != len(lastrow):
        raise TypeError(f'rows and boundary must have equal length! Got len(row): {len(row)}, len(lastrow): {len(lastrow)}, len(boundary): {len(boundary)}.')
    newrow = np.zeros_like(row)
    # i mean technically we could do this with generators and lambdas, but I
    # don't think that would be very readable
    for index in range(len(row)):
        if np.isfinite(boundary[index]):
            newrow[index] = boundary[index]
            # breakpoint()
            continue # boundaries require no further calculation
        # not gonna bother with an else clause here lmaoooo
        # uhhhh what finite difference stencil do we want... how about the
        # most boring one, [-1, 0, 1], 2nd order.
        # Have you ever seen [https://web.media.mit.edu/~crtaylor/calculator.html]?
        # It is quite a lovely tool. Anyhow, since we want to do something with
        # a nonlinear stencil, we can't just directly use Cameron Taylor's
        # outputs. Assuming `s` is a vector containing the stencil and `d` is
        # the order of the derivative, Instead use col[c1...cN] =
        # [[s1^0...sN^0]...[s1^N-1...sN^N-1]]^-1 * col[0...d!...0]
        # where `d!` in the final column vector is placed such that col[d]=d!
        # i.e., extracting the column number corresponding to the order and
        # multiplying by `d!`. Here we can assume `h`, the step size, is 1
        # and all units in `s` are normalized to that, for simplicity.
        xstencil = [xpos[index-1] - xpos[index], 0, xpos[index+1] - xpos[index]] # TODO add a check to make sure we aren't going over a boundary condition somewhere...
        xorder = 2
        torder = 2
        tstencil = [t_lastrow - t_now, t_row - t_now, 0]
        # breakpoint()
        xcoeffs = np.linalg.inv(np.transpose(np.vander(xstencil)))[:,-xorder-1]*math.factorial(xorder) # AI was used to remind of the name of Vandermonde matrices
        tcoeffs = np.linalg.inv(np.transpose(np.vander(tstencil)))[:,-torder-1]*math.factorial(torder)
        d2pdx2 = np.dot(xcoeffs, row[index-1:index+2]) # technically a matrix operation with row and column vectors, but a dot product is an equivalent mathematical operation
        # d2pdx2 is the second deriv of pressure with respect to space
        d2pdt2 = c**2 * d2pdx2 # by definition
        # d2pdt2 = np.dot(tcoeffs, [lastrow[index], row[index], newrow[index])
        # d2pdt2 = np.dot(tcoeffs[:-1], [lastrow[index], row[index]) + tcoeffs[-1]*newrow[index]
        # newrow[index] = (d2pdt2 - np.dot(tcoeffs[:-1], [lastrow[index], row[index]))/tcoeffs[-1]
        newrow[index] = (d2pdt2 - np.dot(tcoeffs[:-1], [lastrow[index], row[index]]))/tcoeffs[-1]
        # breakpoint()
    return newrow


def sim_explicit(tstep, maxt, L_points):
    t_points = int(maxt / tstep + 1)

    xpos = np.linspace(0, L, L_points) # list of x positions
    # technically would could keep this constant, but I've always wanted to make a
    # solver with a nonlinear grid and never quite found the time for it

    tpos = np.arange(0, maxt+tstep, tstep)
    # same deal here, we could just iterate using tstep but nahhhhh lets recalculate
    # dt at each step :)

    initrow = np.full_like(xpos, p_init)
    initrow[0] = 0 # set initial condition (fracture on left end)

    boundary = np.full_like(xpos, np.nan)
    boundary[0] = 0
    boundary[-1] = p_init # set boundary condition, nan means the value is not a
    # a boundary. Sure, I could hard code this, but that would be boring, wouldn't
    # it?

    lastrow = np.full_like(xpos, p_init)
    initrow[0] = 0 # set initial condition (fracture on left end)
    last2row = np.full_like(xpos, p_init) # strady state before t=0
    lasttime = -0*tstep
    last2time = -1*tstep

    fullsim = []
    for time in tpos:
        newrow = lastrow
        if time != lasttime:
            newrow = iterate_explicit(lastrow, last2row, boundary, xpos, time, lasttime, last2time)
            # fullsim.append(newrow)
        else:
            # newrow = lastrow
            lasttime = -1*tstep # dodgy fix
        fullsim.append(newrow)
        last2row = lastrow
        lastrow = newrow
        last2time = lasttime
        lasttime = time
    # print(fullsim)
    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].contourf(xpos/0.3048, tpos, fullsim/6894.757)
    # for index in np.linspace(0, len(tpos)-1, 10):
    #     ax[1].plot(xpos/0.0348, fullsim[int(index)]/6894.757, label=f"t={tpos[int(index)]}")
    # ax[1].legend()
    # ax[0].set_xlabel("Time, seconds")
    # ax[1].set_xlabel("Pressure, PSI")
    # ax[1].set_ylabel("Position, ft")
    # ax[0].set_title(f"Explicit, dt={tstep}, xpoints={L_points}")
    #
    # # fig2, ax2 = plt.subplots(2)
    # # x_mesh, t_mesh = np.meshgrid(xpos, tpos)
    # # ax2[0].contourf(x_mesh, fullsim, t_mesh)
    # # ax2[1].contourf(t_mesh, np.transpose(fullsim), x_mesh)
    # plt.show()
    return np.array(fullsim), xpos, tpos, tstep, L_points

# sim_explicit(0.001, 0.05, 51)

def iterate_implicit(row, lastrow, boundary, xpos, t_now, t_row, t_lastrow):
    # ok and now i'm lazy and gonna do constant order but maybe I'll make a
    # general solution later?
    # dep_stencil = np.array([1, -(2+xstep**2/c**2/tstep**2), 1])
    # indep_stencil = xstep**2/c**2/tstep**2 * np.array([1, -2])
    # Kx=b
    K = np.identity(len(row)) # start with identity, then fill in relations
    b = [] # build this one out
    for index in np.arange(len(row)):
        if np.isnan(boundary[index]):
            xstencil = [xpos[index-1] - xpos[index], 0, xpos[index+1] - xpos[index]] # TODO add a check to make sure we aren't going over a boundary condition somewhere...
            xorder = 2
            torder = 2
            tstencil = [t_lastrow - t_now, t_row - t_now, 0]
            xcoeffs = np.linalg.inv(np.transpose(np.vander(xstencil)))[:,-xorder-1]*math.factorial(xorder) # AI was used to remind of the name of Vandermonde matrices
            tcoeffs = np.linalg.inv(np.transpose(np.vander(tstencil)))[:,-torder-1]*math.factorial(torder)
            # ok and now time for some re-arranging...
            dep_stencil = c**2 * xcoeffs
            dep_stencil[1] -= tcoeffs[-1]
            indep_stencil = tcoeffs[0:-1]
            K[index, index-1:index+2] = dep_stencil
            b.append(np.dot(indep_stencil, [lastrow[index], row[index]])) # i mean this could also be a matrix operation but tbh i'll figure that out later
        else:
            b.append(boundary[index])
        # places where we defined the boundary can be left as the identity in K
        # but we do need to update b
    newrow = np.matmul(np.linalg.inv(K), b)
    return newrow


def sim_implicit(tstep, maxt, L_points, plot_density=False, log=False):
    t_points = int(maxt / tstep + 1)

    xpos = np.linspace(0, L, L_points) # list of x positions
    if log:
        base = 1.1
        offset = 2
        xpos = np.logspace(math.log(offset)/math.log(base), math.log(L+offset)/math.log(base), L_points, base=base) - math.log(offset)/math.log(base)

    # technically would could keep this constant, but I've always wanted to make a
    # solver with a nonlinear grid and never quite found the time for it

    tpos = np.arange(0, maxt+tstep, tstep)
    # same deal here, we could just iterate using tstep but nahhhhh lets recalculate
    # dt at each step :)

    initrow = np.full_like(xpos, p_init)
    initrow[0] = 0 # set initial condition (fracture on left end)

    boundary = np.full_like(xpos, np.nan)
    boundary[0] = 0
    boundary[-1] = p_init # set boundary condition, nan means the value is not a
    # a boundary. Sure, I could hard code this, but that would be boring, wouldn't
    # it?

    lastrow = np.full_like(xpos, p_init)
    initrow[0] = 0 # set initial condition (fracture on left end)
    last2row = np.full_like(xpos, p_init) # strady state before t=0
    lasttime = -0*tstep
    last2time = -1*tstep

    fullsim = []
    for time in tpos:
        newrow = lastrow
        if time != lasttime:
            newrow = iterate_implicit(lastrow, last2row, boundary, xpos, time, lasttime, last2time)
        else:
            lasttime = -1*tstep # dodgy fix
        fullsim.append(newrow)
        last2row = lastrow
        lastrow = newrow
        last2time = lasttime
        lasttime = time
    fullsim = np.array(fullsim)
    # print(fullsim)
    # fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
    # ax[1].set_zorder(2)
    # for index in np.linspace(0, len(tpos)-1, 10):
    #     ax[1].plot(xpos/0.3048, fullsim[int(index)]/6894.757, label=f"t={tpos[int(index)]}")
    #     # pass
    # ax[1].legend()
    # ax[0].set_title(f"Implicit, dt={tstep}, xpoints={L_points}")
    # ax[0].set_ylabel("Time, seconds")
    # ax[1].set_ylabel("Pressure, PSI")
    # ax[1].set_xlabel("Position, ft")
    # if plot_density:
    #     histax.set_ylabel("Grid density")
    #     histax = ax[1].twinx()
    #     histax.hist(xpos[:-1]/0.3048, bins=15, label="grid density", alpha=0.5)
    #     histax.set_zorder(1)
    # conts = ax[0].contourf(xpos/0.3048, tpos, fullsim/6894.757)
    # cbar = fig.colorbar(conts, ax=[ax[0], ax[1]])
    # cbar.ax.set_ylabel("Pressure, PSI")
    return np.array(fullsim), xpos, tpos, tstep, L_points

def makeplot(fullsim, xpos, tpos, tstep, L_points, imporexp, plot_density=False, name=None):
    fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
    ax[1].set_zorder(2)
    for index in np.linspace(0, len(tpos)-1, 10):
        ax[1].plot(xpos/0.3048, fullsim[int(index)]/6894.757, label=f"t={tpos[int(index)]:.5f}")
        # pass
    ax[1].legend()
    ax[0].set_title(f"{imporexp}, dt={tstep}, xpoints={L_points}")
    ax[0].set_ylabel("Time, seconds")
    ax[1].set_ylabel("Pressure, PSI")
    ax[1].set_xlabel("Position, ft")
    if plot_density:
        histax = ax[1].twinx()
        histax.set_zorder(1)
        histax.set_ylabel("Grid density")
        histax.hist(xpos[:-1]/0.3048, bins=15, label="grid density", alpha=0.5)
        histax.set_zorder(1)
    conts = ax[0].contourf(xpos/0.3048, tpos, fullsim/6894.757)
    cbar = fig.colorbar(conts, ax=[ax[0], ax[1]])
    cbar.ax.set_ylabel("Pressure, PSI")
    if name != None:
        fig.savefig("plots/" + name)
        plt.close(fig)
    else:
        plt.show()


fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit")

exit()

# compare explicit and implicit solution
fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_1_explicit_51")
fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, 101)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_1_explicit_101")
fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="1_1_implicit_51")
fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 101)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="1_1_implicit_101")

# compare time steps
fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.0001, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="2_1_explicit_-4")
fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.01, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="2_1_explicit_-2")
fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.0001, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="2_1_implicit_-4")
fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.01, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="2_1_implicit_-2")

# compare runtimes
print("Starting timing tests")
def time_exec(f, step, name):
    num = 100
    start = time.time()
    for i in range(num): f(step, 0.05, 51)
    end = time.time()
    print(f"{name} @ dt={step} took {(end-start)/num*1e+3:.3f} ms (average) to execute.")
time_exec(sim_explicit, 0.0001, "Explicit method")
time_exec(sim_explicit, 0.001, "Explicit method")
time_exec(sim_explicit, 0.01, "Explicit method")
time_exec(sim_implicit, 0.0001, "Implicit method")
time_exec(sim_implicit, 0.001, "Implicit method")
time_exec(sim_implicit, 0.01, "Implicit method")

# showcase nonuniform grid density
fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", plot_density=True,name="4_1_implicit_linear")
fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51, log=True)
makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", plot_density=True, name="4_1_implicit_log")

plt.show()


# you might be thinking "damn these comments look ratchet" so:
# 1. I am sick, let me have some fun and
# 2. I write less ratchet comments when I'm paid for my code
# (also I'm bored of writing my paid code comments)
# (also yes I document my code, it's just part of my though process when I'm
# writing. I have the pre-LLM code to prove it.)