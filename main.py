import numpy as np
import math
from monte_carlo import run_mc
from project import run_is
import matplotlib.pyplot as plt

#to implement later: scheme selection for pin

def compare_estimates(zt, mc, T):
    #plt.fill_between(np.arange(1,21), (zt + 2 * sd_is), (zt - 2 * sd_is),color='b', alpha=.1)
    #print((zt - 2 * sd_is))
    #print((zt + 2 * sd_is))
    plt.style.use('ggplot')
    plt.plot(zt, 'bo')
    plt.plot(mc, 'r+')
    plt.yscale('log')
    plt.savefig('{}year.png'.format(T))
    plt.clf()

def no_ci(zt, mc, T):
    #plt.fill_between(np.arange(1,21), (zt + 2 * sd_is), (zt - 2 * sd_is),color='b', alpha=.1)
    #print((zt - 2 * sd_is))
    #print((zt + 2 * sd_is))
    plt.style.use('ggplot')
    plt.plot(zt, 'bo')
    plt.plot(mc, 'r+')
    plt.yscale('log')
    plt.savefig('{}year_noci.png'.format(T))
    plt.clf()


if __name__ == '__main__':
    T = [1, 3, 5]
    samples_mc = 1000000
    samples_is = 10000

    for t in T:
        print('Year {}-----'.format(t))
        ct, zt, VaR, variance_is = run_is(t, samples_is)
        mc_est, variance_mc = run_mc(t, samples_mc)
        compare_estimates(zt, mc_est, t)
        no_ci(zt, mc_est, t)
        print('Variance Ratio: {}'.format(np.divide(variance_is, variance_mc)))
        print('Variance IS: {}'.format(variance_is))
        print('Variance MC: {}'.format(variance_mc))
