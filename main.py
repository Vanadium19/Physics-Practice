import numpy as np
import matplotlib.pyplot as plt

from dumbbell import Dumbbell, draw_dumbbell
from rotation_quaternions import inverse, multiply, norm, rotate_r

DELTA_T = 0.005
STEPS = 1200
DRAW_EVERY = 2


def quaternion_from_axis_angle(axis: list[float], angle_rad: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half_angle = angle_rad / 2.0
    return np.array([np.cos(half_angle), *(np.sin(half_angle) * axis)])


def normalize_quaternion(Q: np.ndarray) -> np.ndarray:
    return Q / np.sqrt(norm(Q))


def euler_rhs(dumbbell: Dumbbell, omega_body: np.ndarray, torque_body: np.ndarray) -> np.ndarray:
    p_i, q_i, r_i = omega_body
    m_x, m_y, m_z = torque_body

    p_i_dot = (m_x - (dumbbell.C - dumbbell.B) * q_i * r_i) / dumbbell.A
    q_i_dot = (m_y - (dumbbell.A - dumbbell.C) * p_i * r_i) / dumbbell.B
    r_i_dot = (m_z - (dumbbell.B - dumbbell.A) * p_i * q_i) / dumbbell.C
    return np.array([p_i_dot, q_i_dot, r_i_dot])


def quaternion_rhs(Q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    omega_quaternion = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * multiply(Q, omega_quaternion)


def rk2_step(
        dumbbell: Dumbbell,
        omega_body: np.ndarray,
        torque_world: np.ndarray,
        delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    torque_body = rotate_r(inverse(dumbbell.Q), torque_world)
    k1_omega = euler_rhs(dumbbell, omega_body, torque_body)
    k1_quaternion = quaternion_rhs(dumbbell.Q, omega_body)

    omega_mid = omega_body + 0.5 * delta_t * k1_omega
    Q_mid = normalize_quaternion(dumbbell.Q + 0.5 * delta_t * k1_quaternion)

    torque_mid_body = rotate_r(inverse(Q_mid), torque_world)
    k2_omega = euler_rhs(dumbbell, omega_mid, torque_mid_body)
    k2_quaternion = quaternion_rhs(Q_mid, omega_mid)

    omega_next = omega_body + delta_t * k2_omega
    quaternion_next = normalize_quaternion(dumbbell.Q + delta_t * k2_quaternion)
    return omega_next, quaternion_next


def kinetic_energy(dumbbell: Dumbbell, omega_body: np.ndarray) -> float:
    p_i, q_i, r_i = omega_body
    return 0.5 * (
            dumbbell.A * p_i ** 2 +
            dumbbell.B * q_i ** 2 +
            dumbbell.C * r_i ** 2
    )


def angular_momentum_norm(dumbbell: Dumbbell, omega_body: np.ndarray) -> float:
    p_i, q_i, r_i = omega_body
    angular_momentum_body = np.array([
        dumbbell.A * p_i,
        dumbbell.B * q_i,
        dumbbell.C * r_i,
    ])
    return float(np.linalg.norm(angular_momentum_body))


def simulate_dumbbell(
        steps: int = STEPS,
        delta_t: float = DELTA_T,
        animate: bool = True,
) -> tuple[Dumbbell, np.ndarray, np.ndarray, np.ndarray]:
    dumbbell = Dumbbell(l1=0.5, l2=1.0, m1=1.0, m2=0.25)

    # Начальная ориентация и угловая скорость выбраны так,
    # чтобы наблюдалась регулярная прецессия при отсутствии внешних воздействий.
    dumbbell.Q = quaternion_from_axis_angle([1.0, 0.0, 0.0], np.radians(35.0))
    omega_body = np.array([1.6, 0.0, 8.0])

    torque_world = np.zeros(3)
    dumbbell.r_c = np.zeros(3)

    axis_trace = [dumbbell.get_symmetry_axis_tip()]
    energy_history = [kinetic_energy(dumbbell, omega_body)]
    momentum_history = [angular_momentum_norm(dumbbell, omega_body)]

    if animate:
        draw_dumbbell(dumbbell, np.array(axis_trace))
        plt.pause(0.1)

    for step in range(steps):
        omega_body, dumbbell.Q = rk2_step(dumbbell, omega_body, torque_world, delta_t)

        axis_trace.append(dumbbell.get_symmetry_axis_tip())
        energy_history.append(kinetic_energy(dumbbell, omega_body))
        momentum_history.append(angular_momentum_norm(dumbbell, omega_body))

        if animate and (step % DRAW_EVERY == 0 or step == steps - 1):
            draw_dumbbell(dumbbell, np.array(axis_trace))
            plt.pause(delta_t)

    return (
        dumbbell,
        np.array(axis_trace),
        np.array(energy_history),
        np.array(momentum_history),
    )


def main():
    dumbbell, _, energy_history, momentum_history = simulate_dumbbell()

    print(
        'Главные моменты инерции:',
        f'A = {dumbbell.A:.4f}, B = {dumbbell.B:.4f}, C = {dumbbell.C:.4f}'
    )
    print(f'Изменение кинетической энергии: {energy_history[-1] - energy_history[0]:.6e}')
    print(f'Изменение нормы момента импульса: {momentum_history[-1] - momentum_history[0]:.6e}')

    plt.show()


if __name__ == '__main__':
    main()
