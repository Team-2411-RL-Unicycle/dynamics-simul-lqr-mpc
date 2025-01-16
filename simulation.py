import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
import cvxpy as cp
import time as clock

class InvertedPendulum:
    def __init__(self, params=None):
        # Define default parameters
        default_params = {
            "g0": 9.81,
            "mw": 0.346,
            "mp": 0.531,
            "lp": 0.1,
            "lw": 0.18,
            "Ip": 0.002250,
            "Iw": 0.000725,
            "tau_max": 1.0,
        }
        self.params = params if params else default_params
        self.g0 = self.params["g0"]
        self.mw = self.params["mw"]
        self.mp = self.params["mp"]
        self.lp = self.params["lp"]
        self.lw = self.params["lw"]
        self.Ip = self.params["Ip"]
        self.Iw = self.params["Iw"]
        self.m0 = self.g0 * (self.mp * self.lp + self.mw * self.lw)
        self.Inet = self.Ip + self.mp * self.lp**2 + self.Iw + self.mw * self.lw**2

        self.tau_max = self.params["tau_max"]

    def sys_dyn(self, x, tau):
        """System dynamics for the inverted pendulum."""

        # Unpack state and control inputs
        phi, theta, phidot, thetadot = x
        # Compute the non-linear system dynamics
        dx = np.zeros_like(x)

        dx[0] = phidot
        dx[1] = thetadot
        dx[2] = (self.m0 * np.sin(phi) - tau) / self.Inet
        dx[3] = tau / self.Iw - self.m0 * np.sin(phi) / self.Inet
        return dx

    def linearized_matrices(self):
        """Retrieve the linearized A and B matrices at equilibrium."""
        # Define symbolic variables
        phi, theta, phidot, thetadot, tau = sp.symbols("phi theta phidot thetadot tau")
        x = sp.Matrix([phi, theta, phidot, thetadot])
        u = sp.Matrix([tau])

        # Define parameters
        g0, mw, mp, lp, lw, Ip, Iw = (
            self.g0,
            self.mw,
            self.mp,
            self.lp,
            self.lw,
            self.Ip,
            self.Iw,
        )
        m0 = g0 * (mp * lp + mw * lw)
        Inet = Ip + mp * lp**2 + Iw + mw * lw**2

        # Define dynamics
        dx = sp.Matrix(
            [
                phidot,
                thetadot,
                (m0 * sp.sin(phi) - tau) / Inet,
                tau / Iw - m0 * sp.sin(phi) / Inet,
            ]
        )

        # Compute Jacobians
        A = dx.jacobian(x)
        B = dx.jacobian(u)

        # Substitute equilibrium point (x = 0, u = 0)
        equilibrium = {phi: 0, theta: 0, phidot: 0, thetadot: 0, tau: 0}
        A_numeric = A.subs(equilibrium)
        B_numeric = B.subs(equilibrium)

        return np.array(A_numeric).astype(float), np.array(B_numeric).astype(float)

    def linearized_matrices_hardcoded(self):
        """Generate hardcoded Jacobian functions for A(x, u) and B(x, u)."""
        # Define symbolic variables
        phi, theta, phidot, thetadot, tau = sp.symbols("phi theta phidot thetadot tau")
        x = sp.Matrix([phi, theta, phidot, thetadot])
        u = sp.Matrix([tau])

        # Define parameters
        g0, mw, mp, lp, lw, Ip, Iw = (
            self.g0,
            self.mw,
            self.mp,
            self.lp,
            self.lw,
            self.Ip,
            self.Iw,
        )
        m0 = g0 * (mp * lp + mw * lw)
        Inet = Ip + mp * lp**2 + Iw + mw * lw**2

        # Define dynamics
        dx = sp.Matrix(
            [
                phidot,
                thetadot,
                (m0 * sp.sin(phi) - tau) / Inet,
                tau / Iw - m0 * sp.sin(phi) / Inet,
            ]
        )

        # Compute Jacobians
        A = dx.jacobian(x)
        B = dx.jacobian(u)

        print(A)
        print(B)

        # Generate NumPy-compatible functions
        A_func = sp.lambdify((phi, theta, phidot, thetadot, tau), A, modules="numpy")
        B_func = sp.lambdify((phi, theta, phidot, thetadot, tau), B, modules="numpy")

        return A_func, B_func

    def simulate(self, x0, Tnet, control_freq, controller):
        """Simulate the system dynamics using a given controller."""
        t_span = (0, Tnet)
        t_eval = np.linspace(*t_span, int(Tnet * control_freq))

        def constrain_control(tau):
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            return tau

        # Wrapper to add controller to sytem dynamics
        def controlled_dyn(t, x):
            tau = constrain_control(controller.get_control(t, x))
            return self.sys_dyn(x, tau)

        # Solve the system
        sol = solve_ivp(controlled_dyn, t_span, x0, t_eval=t_eval)

        # Return time and solution
        return sol.t, sol.y


class BaseController:
    def __init__(self):
        """Base class for controllers."""
        pass

    def get_control(self, t, x):
        raise NotImplementedError("get_control must be implemented in a derived class")


class PIDController(BaseController):
    def __init__(self, kp, ki, kd, delta_t):
        """PID controller class."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.delta_t = delta_t
        self.prev_error = 0
        self.integral = 0
        self.setpoint = 0

    def get_control(self, t, x):
        """Control on phi."""
        phi, theta, phidot, thetadot = x
        error = phi - self.setpoint
        self.integral += error
        derivative = (error - self.prev_error) / self.delta_t
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class LQRController(BaseController):
    def __init__(self, K):
        self.K = K

    def get_control(self, t, x):
        # slice 0, 2, 3 from x
        x = x[[0, 2, 3]]
        """LQR control input."""
        return -self.K @ x


class MPCController(BaseController):
    def __init__(
        self, A, B, Q, R, Qf=None, N=20, tau_max=1.0, warm_start=True, solver_kwargs={}
    ):
        """
        Model Predictive Controller class.

        Args:
        A (ndarray): Discrete system dynamics matrix x_{k+1} = A x_k + B u_k
        B (ndarray): Discrete input matrix x_{k+1} = A x_k + B u_k
        Q (ndarray): State cost matrix
        R (ndarray): Input cost matrix
        Qf (ndarray): Terminal state cost matrix (default: Q)
        N (int): Prediction horizon (default: 20)
        tau_max (float): Maximum control input (default: 1.0)

        """

        super().__init__()
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Qf = Qf if Qf is not None else Q
        self.tau_max = tau_max
        self.N = N

        self.n = A.shape[0]
        self.m = B.shape[1]

        # Build CVXPY problem structure once
        self._build_problem()

        # Store last solution for warm start optimization
        self.last_x_sol = None
        self.last_u_sol = None

        # Store solver kwargs
        self.solver_kwargs = solver_kwargs

    def _build_problem(self):
        # Create variables for optimization states and inputs
        self.x_var = cp.Variable((self.n, self.N + 1))
        self.u_var = cp.Variable((self.m, self.N))

        # Objective function with terminal cost
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.x_var[:, k], self.Q) + cp.quad_form(
                self.u_var[:, k], self.R
            )
        cost += cp.quad_form(self.x_var[:, self.N], self.Qf)

        # Set x0 as a problem parameter
        self.x0_param = cp.Parameter(shape=self.n, name="x0", value=np.zeros(self.n))

        # Define constraints
        self.constraints = []
        # Initial condition
        self.constraints.append(self.x_var[:, 0] == self.x0_param)
        # Dynamics and input constraints
        for k in range(self.N):
            self.constraints += [
                self.x_var[:, k + 1]
                == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k],
                -self.tau_max <= self.u_var[:, k],
                self.u_var[:, k] <= self.tau_max,
            ]

        # Define/compile the optimization problem
        self.prob = cp.Problem(cp.Minimize(cost), self.constraints)

    def solve_mpc(self, x0, warm_start=True):
        # Set the initial state parameter
        self.x0_param.value = x0

        # Start from last known solution if warm_start is enabled
        if warm_start and self.last_x_sol is not None and self.last_u_sol is not None:
            self.x_var.value = self.last_x_sol
            self.u_var.value = self.last_u_sol

        # Solve the problem
        self.prob.solve(solver=cp.CLARABEL, warm_start=warm_start, **self.solver_kwargs)

        # Store solution
        self.last_x_sol = self.x_var.value
        self.last_u_sol = self.u_var.value

        if self.u_var.value is None:
            print("No solution found!")
            return np.zeros(self.m)

        # Return the first control input
        return self.u_var[:, 0].value

    def get_control(self, t, x):
        # Solve the MPC problem
        x = x[[0, 2, 3]]  # pare down to 3 states of interest
        return self.solve_mpc(x, warm_start=True)
