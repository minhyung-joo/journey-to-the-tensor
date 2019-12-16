import numpy as np

g = 9.8
L = 2
mu = 0.1
INITIAL_THETA = np.pi / 4
INITIAL_VELOCITY = -2

def compute_acceleration(theta, theta_p):
    return -mu * theta_p - (g / L) * np.sin(theta)

def theta(t):
    theta = INITIAL_THETA
    theta_p = INITIAL_VELOCITY
    delta_t = 0.01
    for time in np.arange(0, t, delta_t):
        a = compute_acceleration(theta, theta_p)
        theta += theta_p * delta_t
        theta_p += a * delta_t
    
    return theta

print (theta(10))