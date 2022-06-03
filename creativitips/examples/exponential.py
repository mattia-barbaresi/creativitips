import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats


def show():
    expo = stats.expon
    lambda_ = [0.5, 1, 9]
    a = np.linspace(0, 4, 100)

    for la in lambda_:
        plt.plot(a, expo.pdf(a, scale=1. / la), lw=3, label=r"$\lambda = %.1f$" % la)
        plt.fill_between(a, expo.pdf(a, scale=1. / la), alpha=.33)

    plt.legend()
    plt.ylabel("PDF at $z$")
    plt.xlabel("$z$")
    plt.ylim(0, 1.2)
    plt.title(r"Probability density function of an Exponential random variable; differing $\lambda$")
    plt.show()
