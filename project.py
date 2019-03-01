import numpy as np
import math

T = 5
mu = 0.5
n = 1000

theta = 1. / T * math.ceil(mu * n)

def sample_S(T, mu, n):
    pass
    
def sample_I():
    # call q_i_n and q_n
    # returns T-vector with values between 1 and n. Element is the id of the firm that defaults at that timestep. 0 if no one defaults
    pass
 
###
 
def p_i_n(t, Mt):
    pass
    
def p_n(t, state_B):
    pass
    
def q_i_n(t, state_b):
    pass

def q_n(t, theta, Sn):
    pass  

###

def Z_T():
    pass

    
def D_T(    
    
def run_IS_algorithm():
    # generate event times using poisson
    # for each Sm draw Im
    #
    pass
