import numpy as np
import math

T = 5*12
mu = 0.5
n = 1000

theta = 1. / T * math.ceil(mu * n)

def sample_S(T, mu, n):
    # CPG
    pass

def sample_I():
    # CPG
    # call q_i_n and q_n
    # returns T-vector with values between 1 and n. Element is the id of the
    # firm that defaults at that timestep. 0 if no one defaults
    pass

###

# FDE

def p_i_n(t, Mt):
    pass

def p_n(t, state_B):
    pass

# JCS

def q_i_n(t, state_b):
    pass

def q_n(t, theta, Sn):
    pass

###

# JCS
def I_to_CT(I):
    pass

def Z_T(theta, Sn, I):
    # ADS
    #call DT
    pass

def I_to_M(I):
    M = np.zeros((n, T))
    for i in range(T):
        if I[i] > 0:
            M[I[i], i:] = 1
    return M

def D_T(I):
    # ADS
    # mask = int(I != 0)
    # eqns 26 and 27
    p_n(t, )
    pass

# JCS
def run_IS_algorithm():
    # generate event times using poisson
    # for each Sm draw Im
    # check if its a rare event, if so, generate Z_T and define as Yn
    pass
