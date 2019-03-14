import numpy as np
import math
from tqdm import tqdm


n = 100 # unopt time/it > 3 sec

#p = np.zeros((n, timesteps))
np.random.seed(123)


# Parameters for calculation of p_i_n: Note that c is defined as theta in the paper cited by KG >:(
kappa = np.random.uniform(0.5, 1.5, n)
c = np.random.uniform(0.001, 0.051, n) / 10.  # Note: this adjusts the units of time for c from quarters to months
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

#p_n_mat = np.zeros((n, timesteps))

#first_num = np.array([4 * X_0 * gamma**2 * np.exp(gamma * t) for t in range(timesteps)]).T
#first_denom = np.array([(gamma - kappa + (gamma + kappa) * np.exp(gamma * t)) ** 2 for t in range(timesteps)]).T
#second_num = np.array([2 * kappa * c * (np.exp(gamma * t) - 1) for t in range(timesteps)]).T
#second_denom = np.array([gamma - kappa + (gamma + kappa) * np.exp(gamma * t) for t in range(timesteps)]).T

def monte_carlo_sample():
    def cond_surv_fn(t_start, t_end):
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


def sample_S(theta,T):
    # CPG
    # event times up to T with rate theta
    #print('Sampling S')
    event_times = []
    t = 0
    while len(event_times) < n:
        delta = np.random.exponential(1. / theta)
        t += delta
        if t >= T:
            break
        event_times += [t]
    return event_times

def sample_I(event_times, theta,T):
    # CPG
    # call q_i_n and q_n
    # returns T-vector with values between 1 and n. Element is the id of the firm that defaults at that timestep. 0 if no one defaults

    I = np.full(n, float('inf'))
    for i, Sm in enumerate(event_times):
        if Sm >= T:
            break
        if i == 0:
            prev_state = np.zeros(n)
        else:
            prev_state = generate_Mt(event_times[i-1], I)
        q_i = q_i_n(Sm, prev_state, theta)
        q = q_i / np.sum(q_i)
        transition_idx = np.argmax(np.random.multinomial(1, q))
        I[transition_idx] = Sm
    return I

###

# FDE

# Create an array, of size n (number of firms):
def p_i_n(t, Mt, cached=False):
    #global p_n_mat
    #cached = False
    #if cached:
    #    return p_n_mat[:, t]
    #Mt = np.random.randint(0, 1, 100)
    #t /= 12.
    p_i_n = np.zeros(n)
    third = np.matmul(beta, Mt)
    first_num = 4 * X_0 * gamma**2 * np.exp(gamma * t)
    first_denom = (gamma - kappa + (gamma + kappa) * np.exp(gamma * t)) ** 2
    second_num = 2 * kappa * c * (np.exp(gamma * t) - 1)
    second_denom = gamma - kappa + (gamma + kappa) * np.exp(gamma * t)

    #p_i_n = (first_num[:,t] / first_denom[:,t]) + (second_num[:,t] / second_denom[:,t]) + third
    p_i_n = (first_num / first_denom) + (second_num / second_denom) + third
    p_i_n[np.array_equal(Mt, np.ones(n))] = 0.

    #p_n_mat[:, t] = np.array(p_i_n)
    return p_i_n


def generate_Mt(t, I):
    return (I <= t).astype('float')

# Sum of individual p_i_n (at a given time-step):
def p_n(t, state_B, theta, cached=False):
    rates = p_i_n(t, state_B, cached=cached)
    res = np.sum(rates)
    return res

# JCS

def q_i_n(t, state_b, theta):
    num = p_i_n(t,state_b)
    denom = p_n(t,state_b, theta)
    if denom == 0:
        return num
    else:
        return num / denom * theta


def q_n(t, state_b, theta):
    return np.sum(q_i_n(t, state_b, theta))
###

# JCS

def D_T(I, Sm, theta,T): # We assume I is sorted in increasing default times.
    delta = 0.01
    D = np.log(T * p_n(0, generate_Mt(0, I), theta, cached=True))
    for i in range(len(Sm)):
        if i != 0:
            s_ = Sm[i - 1]
            Ms_ = generate_Mt(s_, I)
            D += np.log(T * p_n(s_, Ms_, theta, cached=True))
    D -= sum(p_n(s, generate_Mt(s, I), theta, cached=True) * delta for s in np.arange(0, T + delta, delta))
    return D


def Z_T(I, Sm, theta,T): # We assume I is sorted in increasing default times.
    CT = np.sum(generate_Mt(T, I))
    D = D_T(I, Sm, theta,T)
    return CT, np.exp(T * theta - (CT * np.log(T * theta)) + D)

# JCS
# generate event times using poisson
# for each Sm draw Im
# check if its a rare event, if so, generate Z_T and define as Yn


def sample_Z(theta,T):
    #global p_n_mat
    #p_n_mat = np.zeros((n, timesteps))
    S = sample_S(theta,T)
    I = sample_I(S, theta,T)
    ct, z = Z_T(I, S, theta,T)
    return ct, z

def gen_bootstrap(distribution):
    num_bootstraps = 100
    estimators = []
    for _ in range(num_bootstraps):
        estimators.append(np.mean(np.random.choice(distribution,len(distribution))))
    #return np.var(estimators)
    return np.percentile(estimators,2.5),np.percentile(estimators,97.5)

def run_is(T,n_samples,):
    mu_ct = []
    mu_zt = []
    VaR = []
    variance_is = []
    for mu in tqdm(np.arange(0.01, 0.21, 0.01)):
        cutoff = mu * n
        samples = []
        counts = []
        theta = (1. / T) * cutoff
        for _ in range(n_samples):
            ct, z = sample_Z(theta,T)
            z = z * (ct >= cutoff)
            samples += [z]
            counts += [ct]
        counts = np.array(counts)
        samples = np.array(samples)
        mu_ct.append(float(np.count_nonzero(counts >= cutoff)) / n_samples)
        mu_zt.append(np.mean(samples))
        VaR.append(np.percentile(samples, 95))
        variance_is.append(gen_bootstrap(samples))

    print("mu_ct = {}".format(mu_ct))
    print("mu_zt = {}".format(mu_zt))
    print("VaR = {}".format(VaR))
    print("variance = {}".format(variance_is))
    return mu_ct, mu_zt, VaR, variance_is


'''samples = []
for _ in tqdm(range(1000)):
    samples += [monte_carlo_sample()]
variance_mc = [np.var(np.array(samples) >= cutoff) for cutoff in range(0, 20)]
distr = [np.mean(np.array(samples) >= cutoff) for cutoff in range(0, 20)]
print(distr)
exit()'''

'''
n_samples = 10
samples = []
for _ in range(n_samples):
    S = sample_S()
    I = sample_I(S)
    samples += [Z_T(S,I)]
print(samples)
'''