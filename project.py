import numpy as np
import math
from tqdm import tqdm

def monte_carlo_sample():
    def cond_surv_fn(t_start, t_end):
        #print((t_start, t_end))
        return np.exp(-sum(p_n(t, M[:, t]) for t in range(t_start, t_end+1)))
    
    def invert_cond_surv_fn(t_prev, u):
        cdf = 0.
        for s in np.arange(1, timesteps - t_prev):
            #print(s)
            cdf += cond_surv_fn(t_prev, t_prev + s)
            #print(cdf)
            if cdf > u:
                return s
    
    M = np.zeros((n, timesteps))
    t_prev = 0
    event_times = []
    while t_prev < timesteps:
        u = np.random.rand()
        inter_arrival = invert_cond_surv_fn(t_prev, u)
        if inter_arrival is None:
            break
        t_prev += inter_arrival
        event_times += [t_prev]
    
    return len(event_times)
    

def sample_S(theta):
    # CPG
    # event times up to T with rate theta
    #print('Sampling S')
    event_times = []
    t = 0
    while len(event_times) < n:
        delta = np.random.exponential(1. / theta)
        t += math.ceil(delta)
        event_times += [t]
    return event_times

def sample_I(event_times, theta):
    # CPG
    # call q_i_n and q_n
    # returns T-vector with values between 1 and n. Element is the id of the firm that defaults at that timestep. 0 if no one defaults

    M = np.zeros((n, timesteps))
    I = np.zeros(timesteps)
    #print('Sampling I')
    for i, Sm in enumerate(event_times):
        if Sm >= timesteps:
            break
        #print((i, Sm))
        if i == 0:
            prev_state = np.zeros_like(M[:, 0])
        else:
            prev_state = M[:, event_times[i-1]]
        q = q_i_n(Sm, prev_state, theta) / q_n(Sm, prev_state, theta)
        transition_idx = np.argmax(np.random.multinomial(1, q))
        M[transition_idx, Sm:] = np.ones_like(M[transition_idx, Sm:])
        I[Sm] = transition_idx + 1
    return I

###

# FDE

p_n_mat = None
# Create an array, of size n (number of firms):
def p_i_n(t, Mt, cached=False):
    global p_n_mat
    if cached:
        return p_n_mat[:, t]
    #Mt = np.random.randint(0, 1, 100)
    #t /= 12.
    p_i_n = np.zeros(n)
    
    first_num = 4 * X_0 * gamma**2 * np.exp(gamma * t)
    first_denom = (gamma - kappa + (gamma + kappa) * np.exp(gamma * t)) ** 2
    second_num = 2 * kappa * c * (np.exp(gamma * t) - 1)
    second_denom = gamma - kappa + (gamma + kappa) * np.exp(gamma * t)
    third = np.matmul(beta, Mt)
    
    p_i_n = (first_num / first_denom) + (second_num / second_denom) + third
    p_i_n[np.array_equal(Mt, np.ones(n))] = 0.
    
    '''
    
    for i in range(n):
        if Mt[i] == 1:
            p_i_n[i] = 0.
            continue
        first_num = 4 * X_0[i] * gamma[i]**2 * np.exp(gamma[i] * t)
        first_denom = (gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t)) ** 2
        second_num = 2 * kappa[i] * c[i] * (np.exp(gamma[i] * t) - 1)
        #second_num = - c[i] * kappa[i] * (kappa[i] ** 2 - gamma[i] ** 2) * (np.exp(gamma[i] * t) - 1)
        second_denom = gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t)
        #second_denom = sigma[i] ** 2 * (gamma[i] - kappa[i] + (gamma[i] + kappa[i]) * np.exp(gamma[i] * t))
        third = 0.
        for j in range(n):
            if i != j:
                third += beta[i, j] * Mt[j]
        p_i_n[i] = (first_num / first_denom) + (second_num / second_denom) + third
    
    '''
    
    p_n_mat[:, t] = np.array(p_i_n)
    return p_i_n


# In[23]:


# Sum of individual p_i_n (at a given time-step):
def p_n(t, state_B, cached=False):
    rates = p_i_n(t, state_B)
    #print(np.sum(rates))
    return np.sum(rates)

# JCS

def q_i_n(t, state_b, theta):
    num = p_i_n(t,state_b)
    denom = p_n(t,state_b)
    if denom == 0:
        return num
    else:
        return num / denom * theta


def q_n(t, state_b, theta):
    return np.sum(q_i_n(t, state_b, theta))
###

# JCS

def I_to_CT(I):
    CT = np.count_nonzero(I)
    return CT

def I_to_M(I):
    M = np.zeros((n, timesteps))
    for i in range(timesteps):
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
    delta = 1
    M = I_to_M(I)
    D = 0
    for s in np.squeeze(np.argwhere(I != 0), axis=1):
        D += np.log(timesteps*p_n(*Ms_minus(s, M), cached=True))
    D -= sum(p_n(int(s), M[:, int(s)], True)*delta for s in np.arange(0, timesteps, delta))
    return D

def Z_T(Sn, I, theta):
    #print('Computing Z_T')
    # ADS
    Z = 1
    CT = I_to_CT(I)
    D = D_T(I)
    return CT, np.exp(timesteps * theta - CT * np.log(timesteps * theta) + D)

# JCS
# generate event times using poisson
# for each Sm draw Im
# check if its a rare event, if so, generate Z_T and define as Yn


def sample_Z(theta):
    global p_n_mat
    p_n_mat = np.zeros((n, timesteps))
    S = sample_S(theta)
    I = sample_I(S, theta)
    ct, z = Z_T(S,I,theta)
    return ct, z

'''
samples = []
for _ in tqdm(range(100)):
    samples += [monte_carlo_sample()]
distr = [np.mean(np.array(samples) >= cutoff) for cutoff in range(0, 20)]
print(distr)
    
exit() 
'''

'''
n_samples = 10
samples = []
for _ in range(n_samples):
    S = sample_S()
    I = sample_I(S)
    samples += [Z_T(S,I)]
print(samples)
'''

discretization = 1
T = 1
timesteps = T * discretization
mu = 0.01
n = 1000 # unopt time/it > 3 sec

p = np.zeros((n, timesteps))


#theta = (1. / timesteps) * math.ceil(mu * n)

# Parameters for calculation of p_i_n: Note that c is defined as theta in the paper cited by KG >:(
kappa = np.random.uniform(0.5, 1.5, n) / 3.0
c = np.random.uniform(0.001, 0.051, n) / 3.0 # Note: this adjusts the units of time for c from quarters to months
sigma_tilde = np.random.uniform(0, 0.2, n)
X_0 = np.array(c) # Make sure to review if correct.

# Construct sigma array:
sigma = np.array([min(np.sqrt(2 * kappa[i] * c[i]), sigma_tilde[i]) for i in range(n)])

# Construct gamma array:
gamma = np.array([np.sqrt(kappa[i]**2 + (2 * sigma[i]**2)) for i in range(n)])

# Beta:
beta = np.random.uniform(0, 0.01, (n, n)) / 10.0 # TODO IS THIS LEGIT?
idx = np.arange(n)
beta[idx, idx] = np.zeros_like(idx)

n_samples = 100 
mu_ct = []
mu_zt = []
for i in tqdm(range(20)):
    mu = 0.01 * (i+1)
    samples = []
    counts = []
    theta = (1. / timesteps) * math.ceil(mu * n)
    for _ in range(n_samples):
        ct, z = sample_Z(theta)
        samples += [z]
        counts += [ct]
    counts = np.array(counts)
    samples = np.array(samples)
    print(1.0 / (np.mean(samples) * T))
    #print(len(np.argwhere(counts>=np.array((mu*n))))/n_samples)
    mu_ct += [len(np.argwhere(counts>=np.array((mu*n))))/n_samples]
    mu_zt += [1.0 / (np.mean(samples) * T)]
print(mu_ct)
print(mu_zt)