import numpy as np
from project import p_i_n, T, n
from tqdm import tqdm

np.random.seed(123)

K = 1

def sim():
    t = 0
    k = 0
    event_times = [0]
    M = np.zeros(n)
    J = np.sum(p_i_n(t, M))
    # J = (0.2 * n) / T
    while t < T and k < n:
        e = np.random.exponential(1 / J)
        if e > K:
            t = t + K
            continue
        else:
            if t + e > T:
                break
            u = np.random.rand()
            h_i = p_i_n(t + e, M)
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
    print('--------------------------- T = {} ------------------------------'.format(T))
    print('n = {}'.format(n))
    dist = np.array([sim() for _ in range(1000000)])
    for mu in tqdm(np.arange(0.01, 0.21, 0.01)):
        print('mu = {:0.2f}: {}'.format(mu, np.mean(dist >= mu * n)))

if __name__ == '__main__':
    main()
