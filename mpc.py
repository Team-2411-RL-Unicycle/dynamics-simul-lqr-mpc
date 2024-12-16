import simulation as sim
import numpy as np
import cvxpy as cp

params = {
    "g0": 9.81,
    "mw": 0.346,
    "mp": 0.531,
    "lp": 0.1,
    "lw": 0.18,
    "Ip": 0.002250,
    "Iw": 0.000725,
    "tau_max": 1.0,
}

x0 = np.array([1, 0, 0, 0])
Tnet = 5  # s
CF = 100  # Hz

pendulum = sim.InvertedPendulum(params=params)


def Afunc(state):
    phi, phidot, thetadot = state
    A = np.array(
        [[0, 1, 0], [58.0587 * np.cos(phi), 0, 0], [-58.0587 * np.cos(phi), 0, 0]]
    )
    return A


def Bfunc(state):
    phi, phidot, thetadot = state
    B = np.array([[0], [-51.294], [1379.31]])
    return B


n = 3  # Number of states
m = 1  # Control dim
T = 20  # Horizon

state_linearization = np.array([0, 0, 0])
A = Afunc(state_linearization)
B = Bfunc(state_linearization)
# Given continuous-time A, B and delta_t
delta_t = 1 / 100.0
A_d = np.eye(n) + delta_t * A
B_d = delta_t * B

w_final = 3.0
Q = np.diag([1e5, 0.1, 0.015])
R = np.diag([100])
Qf = w_final * Q

# Constraints
u_max = 1.0
u_min = -1.0

# Initial condition
x0 = np.array([0.5, 0.0, 0.0])

# # CVXPY Variables
# x = cp.Variable((n, T + 1))  # states from x_0 to x_T
# u = cp.Variable((m, T))  # inputs from u_0 to u_{T-1}

# # Objective
# cost = 0
# for k in range(T):
#     cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
# cost += cp.quad_form(x[:, T], Qf)  # terminal cost

# # Constraints
# constraints = [x[:, 0] == x0]
# for k in range(T):
#     constraints += [
#         x[:, k + 1] == A_d @ x[:, k] + B_d @ u[:, k],
#         u_min <= u[:, k],
#         u[:, k] <= u_max,
#     ]

# # Solve the problem
# prob = cp.Problem(cp.Minimize(cost), constraints)

# # Use Clarabel as solver
# kwargs = {"solver": cp.CLARABEL, "verbose": True, "max_iter": 20, "time_limit": 0.0005}

# prob.solve(**kwargs)


# Example usage:
# Assume A, B, Q, R, Qf are defined and system is linearized at some point
solver_kwargs = {"verbose": True, "max_iter": 20, "time_limit": 0.001}

mpc = sim.MPCController(
    A_d, B_d, Q, R, Qf=Qf, N=20, tau_max=1.0, solver_kwargs=solver_kwargs
)
x0 = np.array([0.5, 0.0, 0.0])
u0 = mpc.solve_mpc(x0)  # First solve, no warm start available

# Next timestep:
x0_next = A_d @ x0 + B_d @ u0  # Just an example, you'd have actual plant measurements
u1 = mpc.solve_mpc(x0_next, warm_start=True)  # Uses last solution as starting point

x1_next = A_d @ x0_next + B_d @ u1
u2 = mpc.solve_mpc(x1_next, warm_start=True)

print(f"First control input: {u0}")
print(f"Second control input: {u1}")
print(f"Third control input: {u2}")