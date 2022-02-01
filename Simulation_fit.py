from scipy.stats import *
import matplotlib.pyplot as plt
import numpy as np


class getParament(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def beta_P(self):
        X = beta.rvs(self.a, self.b, size=1000)
        return X

    def binomial_P(self):
        X = binom(self.a, self.b)
        x = np.arange(0, self.a)
        y = X.pmf(x)
        plt.bar(x, y, width=0.8)
        plt.plot(y)
        plt.title("Binom fit the histogram.")
        plt.legend(["Fitted Distribution",
                    "Observed Data"])
        plt.show()

    def normal_P(self):
        X = norm.rvs(self.a, self.b, 1000)
        return X

    def rayleigh_P(self):
        X = rayleigh.rvs(scale=self.a, size=1000)
        return X

    def f_P(self):
        X = f.rvs(dfn=self.a, dfd=self.b, size=1000)
        return X

    def lognorm_P(self):
        X = lognorm.rvs(loc=self.a, s=self.b, size=100)
        return X

    def cauchy_P(self):
        X = cauchy.rvs(self.a, self.b, size=100)
        return X

    def laplace_P(self):
        X = laplace.rvs(self.a, self.b, 1000)
        return X

    def lomax_P(self):
        X = lomax.rvs(scale=self.a, c=self.b, size=1000)
        return X

    def logistic_P(self):
        X = logistic.rvs(self.a, self.b, 1000)
        return X

    def gamma_P(self):
        X = gamma.rvs(a=self.a, scale=self.b, size=1000)
        return X


class Simulation_Funktion(object):
    def __init__(self, X):
        self.X = X

    # Simulik
    def Sim(self):
        plt.hist(self.X, density=True)
        plt.title("Observed data histogram")
        plt.show()


class fit_Funktion(Simulation_Funktion):
    def __init__(self, X):
        super().__init__(X)

    def beat_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)
        a, b, _, _ = beta.fit(self.X)
        X = beta.rvs(a, b, size=1000)
        W, P = ttest_ind(self.X, X)
        pdf = beta.pdf
        plt.plot(x, pdf(x, a=a, b=b))
        plt.title("Beat fit the histogram.")
        plt.legend(["Fitted Distribution mit\n α=%s\nβ=%s\nP=%s" % (a, b, P),
                    "Observed Data"])
        plt.show()

    # def binomial_Fit(self):
    #     x = np.arange(0, 100)
    #     y = self.X.pmf(x)
    #     plt.bar(x, y, width=0.8)
    #     plt.plot(y)
    #     plt.title("Binom fit the histogram.")
    #     plt.legend(["Fitted Distribution",
    #                 "Observed Data"])
    #     plt.show()

    # def binomial_Sim(self):
    #     self.x = np.arange(0, 100)
    #     y = self.X.pmf(self.x)
    #     plt.bar(self.x, y, width=0.8)
    #     plt.title("Observed data histogram")
    #     plt.show()

    def normal_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        mu, std = norm.fit(self.X)
        X = norm.rvs(loc=mu, scale=std, size=1000)
        W, P = ttest_ind(self.X, X)

        pdf = norm.pdf
        y = pdf(x, loc=mu, scale=std)
        plt.plot(x, y)

        plt.title("Normal fit the histogram.")
        plt.legend(["Fitted Distribution\n mit μ=%s\n und σ² =%s\nW =%s\n P=%s"
                    % (mu, std, W, P),
                    "Observed Data"])
        plt.show()

    def rayleigh_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        _, sigma = rayleigh.fit(self.X)
        X = rayleigh.rvs(scale=sigma, size=1000)
        W, P = ttest_ind(self.X, X)

        pdf = rayleigh.pdf
        plt.plot(x, pdf(x, scale=sigma))
        plt.title("Rayleigh fit the histogram.")
        plt.legend(["Fitted Distribution\n mit σ=%s\nW =%s\n P=%s" % (sigma, W, P),
                    "Observed Data"])
        plt.show()

    def f_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        parament = f.fit(self.X)
        dfn, dfd = parament[0], parament[1]
        X = f.rvs(dfn=dfn, dfd=dfd, size=1000)
        W, P = ttest_ind(self.X, X)
        pdf = f.pdf
        plt.plot(x, pdf(x, dfn=dfn, dfd=dfd))
        plt.title("F fit the histogram.")
        plt.legend(["Fitted Distribution\n mit d1=%s\n d2=%s\nP=%s" % (dfn, dfd, P),
                    "Observed Data"])
        plt.show()

    def gamma_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        a, _, b = gamma.fit(self.X)
        X = gamma.rvs(a=a, scale=b, size=1000)
        W, P = ttest_ind(self.X, X)
        pdf = gamma.pdf
        plt.plot(x, pdf(x, a=a, scale=b))
        plt.title("Gamma fit the histogram.")
        plt.legend(["Fitted Distribution\n mit k=%s\n und θ =%s\nP=%s" % (a, b, P),
                    "Observed Data"])
        plt.show()

    def lognorm_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        parament = lognorm.fit(self.X)
        mu, std = parament[-2], parament[0]
        X = lognorm.rvs(loc=mu, s=std, size=100)
        W, P = ttest_ind(self.X, X)
        pdf = lognorm.pdf
        plt.plot(x, pdf(x, s=std, loc=mu))
        plt.title("Lognormal fit the histogram.")
        plt.legend(["Fitted Distribution\n mit μ=%s\n und σ²=%s\nP=%s" % (mu, std, P),
                    "Observed Data"])
        plt.show()

    def cauchy_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        mu, std = cauchy.fit(self.X)
        X = cauchy.rvs(loc=mu, scale=std, size=100)
        W, P = ttest_ind(self.X, X)
        pdf = cauchy.pdf
        plt.plot(x, pdf(x, loc=mu, scale=std))
        plt.title("Cauchy fit the histogram.")
        plt.legend(["Fitted Distribution\n mit x0=%s\n und γ =%s\nP=%s" % (mu, std, P),
                    "Observed Data"])
        plt.show()

    def laplace_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        mu, std = laplace.fit(self.X)
        X = laplace.rvs(loc=mu, scale=std, size=1000)
        W, P = ttest_ind(self.X, X)
        pdf = laplace.pdf
        plt.plot(x, pdf(x, loc=mu, scale=std))
        plt.title("Laplace fit the histogram.")
        plt.legend(["Fitted Distribution\n mit μ=%s\n und λ=%s\nP=%s" % (mu, std, P),
                    "Observed Data"])
        plt.show()

    def lomax_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)
        parament = lomax.fit(self.X)
        std, c = parament[-1], parament[0]
        X = lomax.rvs(scale=parament[-1], c=parament[0], size=1000)
        W, P = ttest_ind(self.X, X)
        pdf = lomax.pdf
        plt.plot(x, pdf(x, *parament))
        plt.title("Lomax fit the histogram.")
        plt.legend(["Fitted Distribution mit \nλ=%s, \nα=%s\nP=%s" % (std, c, P),
                    "Observed Data"])
        plt.show()

    def logistic_Fit(self):
        x = np.linspace(self.X.min() - 0.01, self.X.max() + 0.01)
        plt.hist(self.X, density=True)

        parament = logistic.fit(self.X)
        X = lognorm.rvs(loc=parament[0], s=parament[-1], size=100)
        W, P = ttest_ind(self.X, X)
        pdf = logistic.pdf
        plt.plot(x, pdf(x, *parament))
        plt.title("logistic fit the histogram.")
        plt.legend(["Fitted Distribution mit \nμ=%s\ns=%s\nP=%s" % (parament[0], parament[1], P),
                    "Observed Data"])
        plt.show()

# if __name__ == '__main__':
#     X = getParament(40, 10).normal_P()
#     fit_Funktion(X).Sim()
