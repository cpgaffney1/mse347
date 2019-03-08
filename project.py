import numpy as np
import math

T = 5 * 12
mu = 0.2
n = 100

theta = 1. / T * math.ceil(mu * n)

# Parameters for calculation of p_i_n: Note that c is defined as theta in the paper cited by KG.
kappa = np.random.uniform(0.5, 1.5, n)
c = np.random.uniform(0.001, 0.051, n)
sigma_tilde = np.random.uniform(0, 0.2, n)
X_0 = np.array(c) # Make sure to review if correct.

# Construct sigma array:
sigma = [min(np.sqrt(2 * kappa[i] * c[i]), sigma_tilde[i]) for i in range(n)]

# Construct gamma array:
gamma = [np.sqrt(kappa[i]**2 + (2 * sigma[i]**2)) for i in range(n)]

# Beta:
beta = np.random.uniform(0, 0.01, (n, n)) / 10. # TODO IS THIS LEGIT?

def sample_S():
    # CPG
    # event times up to T with rate theta
    print('Sampling S')
    event_times = []
    t = 0
    while len(event_times) < n:
        delta = np.random.exponential(1. / theta)
        t += math.ceil(delta)
        event_times.append(t)
    return event_times

def sample_I(event_times):
    # CPG
    # call q_i_n and q_n
    # returns T-vector with values between 1 and n. Element is the id of the firm that defaults at that timestep. 0 if no one defaults

    M = np.zeros((n, T))
    print('Sampling I')
    for i, Sm in enumerate(event_times):
        if Sm > T:
            break
        print((i, Sm))
        if i == 0:
            prev_state = np.zeros_like(M[:, 0])
        else:
            prev_state = M[:, event_times[i-1]]
        q = q_i_n(Sm, prev_state) / q_n(Sm, prev_state)
        transition_idx = np.argmax(np.random.multinomial(1, q))
        M[transition_idx, Sm:] = np.ones_like(M[transition_idx, Sm:])

    I = np.zeros_like(M[0])
    for i in range(M.shape[0]):
        default_time = np.argmax(M[i])
        if default_time != 0:
            I[default_time] = int(i + 1)
    return I

###

# FDE

# Create an array, of size n (number of firms):
def p_i_n(t, Mt):
    #Mt = np.random.randint(0, 1, 100)
    #t /= 12.
    p_i_n = np.zeros(n)
    for i in range(n):
        if Mt[i] == 1:
            p_i_n[i] = 0.
            continue
        first_num = 4 * X_0[i] * gamma[i]**2 * np.exp(gamma[i] * t)
        first_denom = (gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t)) ** 2
        second_num = 2 * kappa[i] * c[i] * (np.exp(gamma[i] * t) - 1)
        second_denom = gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t)
        third = 0.
        for j in range(n):
            if i != j:
                third += beta[i, j] * Mt[j]
        p_i_n[i] = (first_num / first_denom) + (second_num / second_denom) + third
    return p_i_n


# In[23]:


# Sum of individual p_i_n (at a given time-step):
def p_n(t, state_B):
    probs = p_i_n(t, state_B)
    return np.sum(probs)

# JCS

def q_i_n(t, state_b):
    num = p_i_n(t,state_b)
    denom = p_n(t,state_b)
    if denom == 0:
        return num
    else:
        return num / denom * theta


def q_n(t, state_b):
    return np.sum(q_i_n(t, state_b))
###

# JCS

def I_to_CT(I):
    CT = np.count_nonzero(I)
    return CT

def I_to_M(I):
    M = np.zeros((n, T))
    for i in range(T):
        if I[i] > 0:
            M[int(I[i]) - 1, i:] = 1
    return M


def Ms_minus(s, M):
    s_ = max(0, int(s-1e-6))
    return s_, M[:, s_]

def D_T(I):
    # ADS
    # eqns 26 and 27
    # WHAT ARE OUR TIME STEPS, MONTHS???
    delta = 0.5
    M = I_to_M(I)
    D = 0
    for s in np.argwhere(I != 0):
        D += np.log(T*p_n(*Ms_minus(s, M)))
    D -= sum(p_n(int(s), M[:, int(s)])*delta for s in np.arange(0, T, delta))
    return D

def Z_T(Sn, I):
    print('Computing Z_T')
    # ADS
    Z = 1
    CT = I_to_CT(I)
    D = D_T(I)
    print(D)
    #print(CT)
    #print(theta)
    #print(min(S[-1], T))
    return np.exp(min(Sn[-1], T) * theta - CT * np.log(T * theta) + D)

# JCS
# generate event times using poisson
# for each Sm draw Im
# check if its a rare event, if so, generate Z_T and define as Yn
S = sample_S()
I = sample_I(S)

n_samples = 1
samples = []
for _ in range(n_samples):
    samples += [Z_T(S,I)]
print(samples)
