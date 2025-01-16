import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import simulation as sim


def plot_performance(t, y):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot $\phi$ (Pendulum Angle) in the top-left subplot
    axs[0, 0].plot(t, y[0], label=r"$\phi$ (Pendulum Angle)")
    axs[0, 0].set_title(r"$\phi$ (Pendulum Angle)")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Angle [rad]")
    axs[0, 0].grid()
    axs[0, 0].legend()

    # Plot $\theta$ (Wheel Angle) in the top-right subplot
    axs[0, 1].plot(t, y[1], label=r"$\theta$ (Wheel Angle)", color="orange")
    axs[0, 1].set_title(r"$\theta$ (Wheel Angle)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Angle [rad]")
    axs[0, 1].grid()
    axs[0, 1].legend()

    # Plot $\dot{\phi}$ (Pendulum Velocity) in the bottom-left subplot
    axs[1, 0].plot(t, y[2], label=r"$\dot{\phi}$ (Pendulum Velocity)", color="green")
    axs[1, 0].set_title(r"$\dot{\phi}$ (Pendulum Velocity)")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Velocity [rad/s]")
    axs[1, 0].grid()
    axs[1, 0].legend()

    # Plot $\dot{\theta}$ (Wheel Velocity) in the bottom-right subplot
    axs[1, 1].plot(t, y[3], label=r"$\dot{\theta}$ (Wheel Velocity)", color="red")
    axs[1, 1].set_title(r"$\dot{\theta}$ (Wheel Velocity)")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Velocity [rad/s]")
    axs[1, 1].grid()
    axs[1, 1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def run_pid(pendulum, x0, Tnet, CF):
    pid = sim.PIDController(kp=2, ki=0, kd=0.5, delta_t=1 / CF)

    t_pid, y_pid = pendulum.simulate(x0=x0, Tnet=Tnet, control_freq=CF, controller=pid)
    return t_pid, y_pid


def run_lqr(pendulum, x0, Tnet, CF, Q=np.diag([10000, 0.1, 0.015]), R=np.diag([100])):

    A_full, B_full = pendulum.linearized_matrices()
    # Remove theta as a state
    state_indices = [0, 2, 3]
    A = A_full[np.ix_(state_indices, state_indices)]
    B = B_full[np.ix_(state_indices, [0])]  # Only single control input

    # Solve the Algebraic Riccati Equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    # Compute the optimal gain matrix K
    K = np.linalg.inv(R) @ B.T @ P

    print(K)

    lqr = sim.LQRController(K)

    t_lqr, y_lqr = pendulum.simulate(x0=x0, Tnet=Tnet, control_freq=CF, controller=lqr)
    return t_lqr, y_lqr


def run_mpc(
    pendulum: sim.InvertedPendulum,
    x0,
    Tnet,
    CF,
    Q,
    R,
    tau_max=1.0,
    T=20,
    w_final=3.0,
    max_iter=20,
    time_limit=0.001,
):
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

    state_linearization = np.array([0, 0, 0])
    A = Afunc(state_linearization)
    B = Bfunc(state_linearization)
    # Given continuous-time A, B and delta_t
    delta_t = 1 / float(CF)
    A_d = np.eye(n) + delta_t * A
    B_d = delta_t * B

    Qf = w_final * Q

    solver_kwargs = {"verbose": True, "max_iter": max_iter, "time_limit": time_limit}

    mpc = sim.MPCController(
        A_d, B_d, Q, R, Qf=Qf, N=T, tau_max=tau_max, solver_kwargs=solver_kwargs
    )

    t_mpc, y_mpc = pendulum.simulate(x0=x0, Tnet=Tnet, control_freq=CF, controller=mpc)

    return t_mpc, y_mpc


def compare_controllers():
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

    t_pid, y_pid = run_pid(pendulum, x0, Tnet, CF)

    Q1 = np.diag([1e4, 0.1, 25e-3])
    R1 = np.diag([1e3])

    Q2 = np.diag([10000, 0.1, 0.015])
    R2 = np.diag([100])

    t_lqr, y_lqr = run_lqr(pendulum, x0, Tnet, CF, Q=Q1, R=R1)
    t_lqr2, y_lqr2 = run_lqr(pendulum, x0, Tnet, CF, Q=Q1, R=R1)

    t_mpc, y_mpc = run_mpc(
        pendulum,
        x0,
        Tnet,
        CF,
        Q=Q1,
        R=R1,
        tau_max=params["tau_max"],
        T=30,
        w_final=10.0,
        time_limit=0.004,
    )

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot $\phi$ (Pendulum Angle) in the left subplot
    axs[0].plot(t_pid, y_pid[0], label="PID Controller", color="blue")
    # axs[0].plot(t_lqr, y_lqr[0], label="LQR Controller", color="red")
    axs[0].plot(t_lqr2, y_lqr2[0], label="LQR Controller (Q1, R1)", color="green")
    axs[0].plot(t_mpc, y_mpc[0], label="MPC Controller (Q1, R1)", color="purple")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Angle [rad]")
    axs[0].grid()
    axs[0].legend()

    # Plot $\thetadot$ (Wheel speed) in the right subplot
    axs[1].plot(t_pid, y_pid[3], label="PID Controller", color="blue")
    axs[1].plot(t_lqr2, y_lqr2[3], label="LQR Controller (Q1, R1)", color="green")
    axs[1].plot(t_mpc, y_mpc[3], label="MPC Controller (Q1, R1)", color="purple")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Speed [rad/s]")
    axs[1].grid()
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save in high resolution
    dir = os.path.dirname(__file__)
    save_path = os.path.join(dir, "../figures/controller_comparison.png")
    plt.savefig(save_path, dpi=300)

    plt.show()


def compare_mpc_settings():
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

    Q1 = np.diag([1e4, 0.1, 25e-3])
    R1 = np.diag([1e3])

    T_vals = [20, 50, 80]
    t_limits = [2e-3, 5e-3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot for different T values
    for T in T_vals:
        t_mpc, y_mpc = run_mpc(
            pendulum,
            x0,
            Tnet,
            CF,
            Q=Q1,
            R=R1,
            tau_max=params["tau_max"],
            T=T,
            w_final=10.0,
            time_limit=0.001,
        )
        axs[0].plot(t_mpc, y_mpc[0], label=f"T={T}")
        axs[1].plot(t_mpc, y_mpc[3], label=f"T={T}")

    # Plot for different time limits
    for t_lim in t_limits:
        t_mpc, y_mpc = run_mpc(
            pendulum,
            x0,
            Tnet,
            CF,
            Q=Q1,
            R=R1,
            tau_max=params["tau_max"],
            T=30,
            w_final=10.0,
            time_limit=t_lim,
        )
        axs[0].plot(t_mpc, y_mpc[0], label=rf"T={30}, $t_{{\mathrm{{lim}}}}$={t_lim}")
        axs[1].plot(t_mpc, y_mpc[3], label=rf"T={30}, $t_{{\mathrm{{lim}}}}$={t_lim}")

    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Angle [rad]")
    axs[0].grid()
    axs[0].legend()

    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Speed [rad/s]")
    axs[1].grid()
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    dir = os.path.dirname(__file__)
    figures_path = os.path.join(dir, "../figures")
    os.makedirs(figures_path, exist_ok=True)
    save_path = os.path.join(dir, "../figures/mpc_settings_comparison.png")
    plt.savefig(save_path, dpi=300)

    plt.show()

if __name__ == "__main__":
    # compare_controllers()
    compare_mpc_settings()
