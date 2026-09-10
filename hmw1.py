import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.ndimage as spn

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

def iterate_implicit(rows, boundary, boundx, xposs, t_now, t_rows, optimize_xpos=False, plotting=False):
    # alrughty so expected inputs
    # rows is the last n rows of the simulation, where n is the order. rows[0]
    # is the oldest, and rows[n-1] is the newest. the output of this function
    # would be rows[n]
    # boundary is the boundary conditions, and boundx is the locations of the
    # boundary values
    # xposs is the x positions represented by rows, same order as rows
    # t_now is the time to calculate
    # t_rows is the timestamps of the rows (rows[0] occurs at t_rows[0])
    # returns newrow (new set of pressure values) and the xpos of that new row
    # this function will try to optimize the distribution of xpos on each
    # iteration, while keeping the number of xpoints the same

    newxpos = xposs[-1]
    if optimize_xpos:
        first_deriv = np.gradient(rows[-1])/np.gradient(xposs[-1])
        second_deriv = np.gradient(first_deriv)/np.gradient(xposs[-1])
        third_deriv = np.gradient(second_deriv)/np.gradient(xposs[-1])
        cdf = np.cumsum(np.abs(third_deriv) * np.hstack(([0], np.diff(xposs[-1]))))
        cdf_blur = spn.gaussian_filter1d(cdf, 30 * c * (t_now - t_rows[-1]), mode='nearest')
        # print(f"Gaussian sigma: {30 * c * (t_now - t_rows[-1])}")
        if cdf_blur[-1] != np.max(cdf_blur):
            breakpoint()
        if cdf_blur[-1] != 0:
            cdf_blur = cdf_blur - cdf_blur[0]
            cdf_blur = cdf_blur + np.linspace(0, cdf_blur[-1]*3.0, len(xposs[-1])) # introduce a little slope to encourage allocation of points to otherwise "unsimulated" regions
            cdf_blur = cdf_blur / cdf_blur[-1] # normalize
            cdf = cdf - cdf[0]
            cdf = cdf + np.linspace(0, cdf[-1]*0.5, len(xposs[-1])) # introduce a little slope to encourage allocation of points to otherwise "unsimulated" regions
            cdf = cdf / cdf[-1] # normalize
            newxpos = np.interp(np.linspace(0, 1, len(xposs[-1])), cdf_blur, xposs[-1]) # inverse interpolation
            newxpos[0] = xposs[-1][0]
            newxpos[-1] = xposs[-1][-1] # force these to be equal
        else:
            # breakpoint()
            newxpos = xposs[-1] # if we can't find a new spacing, just use the old one

    newboundary = np.interp(newxpos, boundx, boundary)
    # Kx=b
    K = np.identity(len(rows[-1])) # start with identity, then fill in relations
    b = [] # build this one out
    for index in np.arange(len(newxpos)):
        if np.isnan(newboundary[index]):
            # need to find n (start with 3) points in previous row that are
            # closest to our point of interest
            xpos_current = newxpos[index]
            indices = np.sort(np.argsort(np.abs(xposs[-1] - xpos_current))[:3]) # i mean there is a faster way to do this but i'm lazy'
            xstencil = [newxpos[index-1], newxpos[index], newxpos[index+1]] - xpos_current # interestringly, this resolves the out-of-bounds issue by only ever picking stencil points that are in bounds
            xorder = 2
            torder = 2
            # tstencil = [t_lastrow - t_now, t_row - t_now, 0]
            tstencil = np.hstack((t_rows[-2:], t_now)) - t_now
            xcoeffs = np.linalg.inv(np.transpose(np.vander(xstencil)))[:,-xorder-1]*math.factorial(xorder) # AI was used to remind of the name of Vandermonde matrices
            tcoeffs = np.linalg.inv(np.transpose(np.vander(tstencil)))[:,-torder-1]*math.factorial(torder)
            # ok and now time for some re-arranging...
            dep_stencil = c**2 * xcoeffs
            dep_stencil[1] -= tcoeffs[-1]
            indep_stencil = tcoeffs[0:-1]
            K[index, index] = 0 # zero this out, identity was just a placeholder
            try:
                K[index, index-1:index+2] = dep_stencil
            except:
                breakpoint()
            # K[index, np.array(indices)] = dep_stencil # output (row) is equal to linear combination of inputs (col)
            b.append(np.dot(indep_stencil, [np.interp(xpos_current, xposs[-2], rows[-2]), np.interp(xpos_current, xposs[-1], rows[-1])])) # TODO there's gotta be a better way to do this...
        else:
            b.append(newboundary[index])
        # places where we defined the boundary can be left as the identity in K
        # but we do need to update b
    # print(f"{K=}")
    # print(f"{b=}")
    # print(f"{xposs=}")
    # print(f"{newxpos=}")
    try:
        newrow = np.matmul(np.linalg.inv(K), b)
    except:
        newrow = np.full_like(rows[-1], np.nan)
        print("oopsie!")
        breakpoint()

    if plotting:
        fig, ax = plt.subplots(3)
        ax[0].plot(xposs[-1], rows[-1], label="0th deriv")
        if optimize_xpos:
            ax[0].plot(xposs[-1], first_deriv, label="1st deriv")
            ax[0].plot(xposs[-1], second_deriv, label="2nd deriv")
            ax[0].plot(xposs[-1], third_deriv, label="3rd deriv")
            ax[1].plot(xposs[-1], cdf, label="cdf")
            ax[1].plot(xposs[-1], cdf_blur, label="cdf blur")
        ax[0].legend()
        ax[1].scatter(newxpos, np.full_like(newxpos, 0.6), label="newxpos")
        ax[1].scatter(xposs[-1], np.full_like(newxpos, 0.4), label="oldxpos")
        ax[1].hist(newxpos, label="histogram", alpha=0.5, density=True, cumulative=True)
        ax[1].legend()
        ax[1].set_ylim(0, 1)
        ax[2].plot(xposs[-1], rows[-1], label = "t-1")
        ax[2].plot(newxpos, newrow, label = "t")
        ax[2].legend()
        ax[0].set_title("optimized" if optimize_xpos else "not optimized")
        plt.show()

    return newrow, newxpos

def sim_implicit(tstep, maxt, L_points, plot_density=False, log=False, adaptive=False):
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

    xposs = [xpos, xpos]
    t_rows = [last2time, lasttime]

    fullsim = [last2row, lastrow]
    for time in tpos:
        if time != lasttime:
            newrow, newxpos = iterate_implicit(fullsim, boundary, xpos, xposs, time, t_rows, optimize_xpos=adaptive, plotting=False)
            fullsim.append(newrow)
            xposs.append(newxpos)
            t_rows.append(time)
    fullsim = np.array(fullsim)
    return np.array(fullsim), xposs, t_rows, tstep, L_points

def makeplot(fullsim, xpos, tpos, tstep, L_points, imporexp, plot_density=None, name=None):
    xpos = np.array(xpos)
    # print(np.shape(xpos))
    # print(np.shape(tpos))
    # print(np.shape(fullsim))
    # print(xpos)
    # print(tpos)
    # print(fullsim)
    # print('\n\n')
    if np.size(xpos[0]) == 1:
        xpos, tpos = np.meshgrid(xpos, tpos)
    else:
        _, tpos = np.meshgrid(xpos[0], tpos) # xpos already meshgrid-friendly
    # print(np.shape(xpos))
    # print(np.shape(tpos))
    # print(np.shape(fullsim))
    fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
    ax[1].set_zorder(2)
    for index in np.linspace(0, len(tpos)-1-1e-6, 10):
        ax[1].plot(xpos[int(np.floor(index))]/0.3048, fullsim[int(np.floor(index))]/6894.757, label=f"t={tpos[int(np.floor(index)), 0]:.5f}")
        # pass
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

## Some adaptive meshing tests and demos:
# L_points = 11
# base = 1.1
# offset = 2
# xpos = np.logspace(math.log(offset)/math.log(base), math.log(L+offset)/math.log(base), L_points, base=base) - offset
# init = 10 * np.exp(-(xpos - 10)**2 / 50)
# boundary = np.full_like(xpos, np.nan)
# boundary[0] = init[0]
# boundary[-1] = init[-1]
# row_opt, xpos_opt = iterate_implicit([init, init], boundary, xpos, [xpos, xpos], 0.002, [0.00, 0.001], plotting=True, optimize_xpos=True)
# row_nopt, xpos_nopt = iterate_implicit([init, init], boundary, xpos, [xpos, xpos], 0.002, [0.00, 0.001], plotting=True, optimize_xpos=False)
# fig, ax = plt.subplots(1)
# ax.plot(xpos_opt, row_opt, label="opt")
# ax.plot(xpos_nopt, row_nopt, label="nopt")
# ax.legend()
# plt.show()


fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, int(100/0.001/1125+1))
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_1_stable")
fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, int(100/0.001/1125+2))
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_2_marginal")
fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, int(100/0.001/1125+3))
makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_3_unstable")

#
# # compare explicit and implicit solution
# fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_1_explicit_51")
# fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.001, 0.05, 101)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="1_1_explicit_101")
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="1_1_implicit_51")
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 101)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="1_1_implicit_101")
# print("finished 1: explicit v implicit")
#
# # compare time steps
# fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.0001, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="2_1_explicit_-4")
# fullsim, xpos, tpos, tstep, L_points = sim_explicit(0.01, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Explicit", name="2_1_explicit_-2")
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.0001, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="2_1_implicit_-4")
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.01, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", name="2_1_implicit_-2")
# print("finished 2: tstep variation")
#
# # compare runtimes
# print("Starting timing tests")
# def time_exec(f, step, name):
#     num = 100
#     start = time.time()
#     for i in range(num): f(step, 0.05, 51)
#     end = time.time()
#     print(f"{name} @ dt={step} took {(end-start)/num*1e+3:.3f} ms (average) to execute.")
# time_exec(sim_explicit, 0.0001, "Explicit method")
# time_exec(sim_explicit, 0.001, "Explicit method")
# time_exec(sim_explicit, 0.01, "Explicit method")
# time_exec(sim_implicit, 0.0001, "Implicit method")
# time_exec(sim_implicit, 0.001, "Implicit method")
# time_exec(sim_implicit, 0.01, "Implicit method")
#
# # showcase nonuniform grid density
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", plot_density="overlay",name="4_1_implicit_linear")
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51, log=True)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "Implicit", plot_density="overlay", name="4_1_implicit_log")
#
# # showcase adaptive meshing
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51, adaptive=False, log=True)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "nonadaptive", plot_density="overlay", name="4_2_implicit_nonadaptive")
# fullsim, xpos, tpos, tstep, L_points = sim_implicit(0.001, 0.05, 51, adaptive=True, log=True)
# makeplot(fullsim, xpos, tpos, tstep, L_points, "adaptive", plot_density="overlay", name="4_2_implicit_adaptive")
# # tbh it does not work very well, but it was still cool to play around with!
#
# plt.show()
#
#
# # you might be thinking "damn these comments look ratchet" so:
# # 1. I am sick, let me have some fun and
# # 2. I write less ratchet comments when I'm paid for my code
# # (also I'm bored of writing my paid code comments)
# # (also yes I document my code, it's just part of my though process when I'm
# # writing. I have the pre-LLM code to prove it.)