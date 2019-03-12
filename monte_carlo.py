import numpy as np

discretization = 12
T = 1
timesteps = T * discretization

n=100

# Parameters for calculation of p_i_n: Note that c is defined as theta in the paper cited by KG >:(
kappa = np.random.uniform(0.5, 1.5, n) # / 4
c = np.random.uniform(0.001, 0.051, n) / 14.0 # Note: this adjusts the units of time for c from quarters to months
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

p_n_mat = np.zeros((n, timesteps))

first_num = np.array([4 * X_0 * gamma**2 * np.exp(gamma * t) for t in range(timesteps)]).T
first_denom = np.array([(gamma - kappa + (gamma + kappa) * np.exp(gamma * t)) ** 2 for t in range(timesteps)]).T
second_num = np.array([2 * kappa * c * (np.exp(gamma * t) - 1) for t in range(timesteps)]).T
second_denom = np.array([gamma - kappa + (gamma + kappa) * np.exp(gamma * t) for t in range(timesteps)]).T


# Create an array, of size n (number of firms):
def p_i_n(t, Mt, cached=False):
    global p_n_mat
    if cached:
        return p_n_mat[:, t]
    #Mt = np.random.randint(0, 1, 100)
    #t /= 12.
    p_i_n = np.zeros(n)
    third = np.matmul(beta, Mt)
    p_i_n = (first_num[:,t] / first_denom[:,t]) + (second_num[:,t] / second_denom[:,t]) + third
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

K = 1

def sim():
    t = 0
    k = 0
    event_times = [0]
    M = np.zeros(n)
    J = np.sum(p_i_n(t, M))
    while t < timesteps and k < n:
        e = np.random.exponential(1 / J)
        if e > K:
            t = t + K
            continue
        else:
            if t + e > timesteps:
                break
            u = np.random.rand()
            h_i = p_i_n(int(t + e), M)
            H = np.sum(h_i)
            J = max(J, H)
            if u * J < H:
                k += 1
                t += e
            else:
                t += e
                continue
        probs = h_i / H
        firm_idx = np.argmax(np.random.multinomial(1, probs))
        M[firm_idx] = 1
    return np.sum(M)


def main():
    dist = np.array([sim() for _ in range(10000)])
    for mu in np.arange(0.01, 0.21, 0.01):
        print('mu = {:0.2f}: {}'.format(mu, np.mean(dist > mu * n)))

if __name__ == '__main__':
    main()
