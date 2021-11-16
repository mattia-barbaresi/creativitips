# samples taken from:
# https://nbviewer.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb

from examples import bayes, poisson, exponential


def run():
    # bayes update rule
    bayes.show()

    # poisson dist. (discrete)
    poisson.show()

    # poisson dist. (continuous)
    poisson.show()


if __name__ == "__main__":
    run()

