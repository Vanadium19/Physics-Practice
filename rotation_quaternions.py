import numpy as np

def multiply(q1, q2):
    a1, a2 = q1[0], q2[0]
    v1 = np.array([q1[1], q1[2], q1[3]])
    v2 = np.array([q2[1], q2[2], q2[3]])
    a = a1*a2 - np.dot(v1, v2)
    v = a1*v2 + a2*v1 + np.cross(v1, v2)
    return np.array([a, v[0], v[1], v[2]])

def rotate(Q, airplane):
    Q_inverse = np.array([Q[0], -Q[1], -Q[2], -Q[3]])
    result = np.zeros(airplane.shape)
    for i in range(airplane.shape[1]):
        r_0 = np.array([0, airplane[0][i], airplane[1][i], airplane[2][i]])
        r = multiply(multiply(Q, r_0), Q_inverse)
        result[0][i] = r[1]
        result[1][i] = r[2]
        result[2][i] = r[3]
    return result

def rotate_r(Q, r):
    Q_inverse = np.array([Q[0], -Q[1], -Q[2], -Q[3]])
    r_0 = np.array([0, r[0], r[1], r[2]])
    r = multiply(multiply(Q, r_0), Q_inverse)
    return np.array([r[1], r[2], r[3]])

def norm(Q):
    return Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3]

def inverse(Q):
    return np.array([Q[0], -Q[1], -Q[2], -Q[3]])