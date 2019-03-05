import numpy as np
import math

T = 5 * 12
mu = 0.01
n = 1000

theta = 1. / T * math.ceil(mu * n)

# Parameters for calculation of p_i_n: Note that c is defined as theta in the paper cited by KG.
kappa = np.random.uniform(0.5, 1.5, n)
c = np.random.uniform(0.001, 0.051, n)
sigma_tilde = np.random.uniform(0, 0.2, n)
X_0 = c # Make sure to review if correct.

# Construct sigma array:
sigma = [0 for x in range(n)]
for i in range(n):
    sigma[i] = min(np.sqrt(2 * kappa[i] * c[i]), sigma_tilde[i])

# Construct gamma array:
gamma = [0 for x in range(n)]
for i in range(n):
    gamma[i] = np.sqrt(kappa[i]**2 + (2 * sigma[i]**2))

# Beta:
beta = np.random.uniform(0, 0.01, (n, n)) / 10.0

def sample_S():
    # CPG
    # event times up to T with rate theta
    event_times = []
    t = 0
    while True:
        delta = np.random.exponential(1. / theta)
        print(delta)
        t += math.ceil(delta)
        if t > T:
            break
        event_times.append(t)
    print(event_times)
    return event_times

def sample_I():
    # CPG
    # call q_i_n and q_n
    # returns T-vector with values between 1 and n. Element is the id of the firm that defaults at that timestep. 0 if no one defaults
    event_times = sample_S()
    M = np.zeros(n, T)

    for i, Sm in enumerate(event_times):
        if i == 0:
            prev_state = np.zeros_like(M[:, 0])
        else:
            prev_state = M[:, event_times[i-1]]
        q = [q_i_n(Sm, prev_state) / q_n(Sm, prev_state) for _ in range(n)]
        transition_idx = np.argmax(np.random.multinomial(1, q))
        M[transition_idx, Sm:] = np.ones_like(M[transition_idx, Sm:])      
    transitions = np.argmax(M != 0, axis=0)
    print(transitions)
    #JCS: I believe the return statement below was missing.
    return event_times, transitions



###

# FDE

# Create an array, of size n (number of firms):
def p_i_n(t, Mt):
    p_i_n = [0 for x in range(n)]
    for i in range(n):
        first_num = 4 * X_0[i] * gamma[i]**2 * np.exp(gamma[i] * t)
        first_denom = gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t)**2
        second_num = 2 * kappa[i] * c[i] * (np.exp(gamma[i] * t) - 1)
        second_denom = gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t)
        third = [0 for x in range(n * n)]
        for j in range(n):
            if i != j:
                third[i * j] = beta[i, j] * Mt[j]
            else:
                third[i * j] = 0
        p_i_n[i] = (first_num / first_denom) + (second_num / second_denom) + np.sum(third)
    return(p_i_n)


# In[23]:


# Sum of individual p_i_n (at a given time-step):
def p_n(t, state_B):
    probs = p_i_n(t, state_B)
    return(np.sum(probs))

# JCS

def q_i_n(t, state_b):
    num = p_i_n(t,state_b)
    denom = p_n(t,state_b)
    if num == 0 and denom == 0:
        return 0
    return (num/denom)*theta


def q_n(t, Sn):
    if Sn>t:
        return theta
    else:
        return 0

###

# JCS

def I_to_CT(I):
    CT = [np.count_nonzero(I[:i]) for i in range(len(I))]
    CT = CT[1:]
    return np.array(CT)

def I_to_M(I):
    M = np.zeros((n, T))
    for i in range(T):
        if I[i] > 0:
            M[I[i], i:] = 1
    return M


def Ms_minus(s, M):
    s_ = max(0, int(s-1e-6))
    return s_, M[:, s_]

def D_T(I):
    # ADS
    # eqns 26 and 27
    # WHAT ARE OUR TIME STEPS, MONTHS???
    delta = 1e-2
    M = I_to_M(I)
    D = 0
    for s in np.argwhere(I != 0):
        D += np.log(T*p_n(*Ms_minus(s, M)))
    D -= sum(p(int(s), M[:, int(s)])*delta for s in np.arange(0, T + delta, delta))
    return D

def Z_T(Sn, I):
    # ADS
    Z = 1
    CT = I_to_CT(I)
    # for i in range(1, n+1):
    #     s = np.argwhere(M[i] == 1)[0]
    #     x = np.log(p_i_n(*Ms_minus(s, M))/q_i_n(*Ms_minus(s, M)))
    #     x -=
    # WHAT IS THE MIN STATEMENT SUPPOSSED TO BE???
    return np.exp(min(np.min(S), T) * theta - CT * np.log(T * theta) + D_T(I))

# JCS
def run_IS_algorithm():
    # generate event times using poisson
    # for each Sm draw Im
    # check if its a rare event, if so, generate Z_T and define as Yn
    S,I = sample_I()
    return np.mean(Z_T(S,I))
