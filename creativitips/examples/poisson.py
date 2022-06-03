from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats


def show():
    figsize(12.5, 4)
    poi = stats.poisson
    lambda_ = [1.5, 10, 20]
    a = np.arange(max(lambda_) * 2)

    for la in lambda_:
        plt.bar(a, poi.pmf(a, la), label="$\lambda = %.1f$" % la, alpha=0.60, lw="3")

    plt.xticks(a + 0.4, a)
    plt.legend()
    plt.ylabel("probability of $k$")
    plt.xlabel("$k$")
    plt.title("Probability mass function of a Poisson random variable; differing $\lambda$ values")
    plt.show()
