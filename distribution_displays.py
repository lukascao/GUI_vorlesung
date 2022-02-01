import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
import math

# -- -- #


def Beta_Distribution(sp1, sp2, *args, **kwargs):
    alpha_list = list(map(float, sp1))
    beta_list = list(map(float, sp2))

    x = np.linspace(0, 1, 100)
    labels = []
    plt.figure(dpi=150)
    for a, b in zip(alpha_list, beta_list):
        labels.append('α={}, β={}'.format(a, b))
        y = beta(a=a, b=b).pdf(x)
        plt.axis([0, 1.0, 0, 2.5])
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.legend(labels=labels, loc='best')
    plt.title('Beta distribution')
    plt.show()


def Binomial_Distribution(sp1, sp2, *args, **kwargs):
    N_list = list(map(float, sp1))
    p_list = list(map(float, sp2))

    m = max(N_list)
    k = np.arange(0, m + 1)

    labels = []
    plt.figure(dpi=150)
    for p, n in zip(p_list, N_list):
        labels.append('p={}, n={}'.format(p, n))
        y = binom.pmf(k, n=n, p=p)

        plt.tick_params(axis='both', labelsize=10)
        plt.scatter(k, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Binomial distribution')
    plt.xlabel('N')
    plt.ylabel('Frequency')
    plt.show()


def Normal_Distribution(sp1, sp2, *args, **kwargs):
    mu_list = list(map(float, sp1))
    sigma_list = list(map(float, sp2))

    m = max(mu_list)
    n = min(mu_list)

    x = np.linspace(n - 10, m + 10, 1000)
    # x-axis range is based on the mu parameter, so it's hard to fits every multiple sets of parameters
    labels = []
    plt.figure(dpi=150)
    for mu, sigma in zip(mu_list, sigma_list):
        labels.append('μ={}, σ²={}'.format(mu, sigma))
        y = norm.pdf(x, loc=mu, scale=math.sqrt(sigma))
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Normal distribution')
    plt.ylabel('Frequency')
    plt.xlabel('x')
    plt.show()


def Poisson_Distribution(sp1, *args, **kwargs):
    mu_list = list(map(float, sp1))

    x = np.arange(20)
    labels = []
    plt.figure(dpi=150)
    for mu in mu_list:
        labels.append('μ={}'.format(mu))
        y = poisson.pmf(x, mu=mu)
        plt.tick_params(axis='both', labelsize=10)
        plt.scatter(x, y)
        plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Poisson distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()


def Rayleigh_Distribution(sp1, *args, **kwargs):
    sigma_list = list(map(float, sp1))

    x = np.linspace(0, 11, 1000)
    labels = []
    plt.figure(dpi=150)
    for sigma in sigma_list:
        labels.append('σ={}'.format(sigma))
        y = rayleigh.pdf(x, scale=sigma)
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Rayleigh distribution')
    plt.show()


def F_Distribution(sp1, sp2, *args, **kwargs):
    d1_list = list(map(float, sp1))
    d2_list = list(map(float, sp2))

    x = np.linspace(0, 5, 1000)
    labels = []
    plt.figure(dpi=150)
    for d1, d2 in zip(d1_list, d2_list):
        labels.append('d1={}, d2={}'.format(d1, d2))
        y = f.pdf(x, dfn=d1, dfd=d2)
        plt.axis([0, 5, 0, 2.5])
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.legend(labels=labels, loc='best')
    plt.title('F distribution')
    plt.show()


def Gamma_Distribution(sp1, sp2, *args, **kwargs):
    k_list = list(map(float, sp1))
    a_list = list(map(float, sp2))

    x = np.linspace(0, 20, 10000)
    labels = []
    plt.figure(dpi=150)
    for k, a in zip(k_list, a_list):
        labels.append('k={}, θ={}'.format(k, a))
        y = gamma.pdf(x, a=k, scale=a)
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Gamma distribution')
    plt.xlabel('x')
    plt.show()


def Geometric_Distribution(sp1, *args, **kwargs):
    p_list = list(map(float, sp1))

    x = np.arange(15)
    labels = []
    plt.figure(dpi=150)
    for p in p_list:
        labels.append('p={}'.format(p))
        y = geom.pmf(x, p=p)
        plt.tick_params(axis='both', labelsize=10)
        plt.scatter(x, y)
        plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Geometric  distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()


def Lognorm_Distribution(sp1, sp2, *args, **kwargs):
    mu_list = list(map(float, sp1))
    sigma_list = list(map(float, sp2))
    m = min(mu_list)
    n = max(mu_list)
    x = np.linspace(m, m + n + 5, 1000)
    labels = []
    plt.figure(dpi=150)
    for mu, sigma in zip(mu_list, sigma_list):
        labels.append('μ={}, σ={}'.format(mu, sigma))
        y = lognorm.pdf(x, s=sigma, loc=mu)
        # plt.axis([0, 2.5, 0, 2.0])
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('log-normal distribution')
    # plt.xlabel('x')
    # plt.ylabel('Frequency')
    plt.show()


def Chi2_Distribution(sp1, *args, **kwargs):  # Chi-Square Distribution
    df_list = list(map(float, sp1))
    m = max(df_list)
    x = np.linspace(0, m * 2, 100)
    labels = []
    plt.figure(dpi=150)
    for df in df_list:
        labels.append('df={}'.format(df))
        y = chi2.pdf(x, df)
        plt.axis([0, m * 2, 0, 0.5])
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.legend(labels=labels, loc='best')
    plt.title('Chi-square distribution')
    plt.ylabel('Frequency')
    plt.xlabel('x')
    plt.show()


def Cauchy_Distribution(sp1, sp2, *args, **kwargs):
    x0_list = list(map(float, sp1))
    gamma_list = list(map(float, sp2))
    m = max(x0_list)
    n = min(x0_list)
    x = np.linspace(n - 5, m + 5, 1000)
    labels = []
    plt.figure(dpi=150)
    for x0, gamma in zip(x0_list, gamma_list):
        labels.append('$x_0$={}, γ={}'.format(x0, gamma))
        y = cauchy.pdf(x, loc=x0, scale=gamma)
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.xlim(-5, 5)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Cauchy distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()


def Laplace_Distribution(sp1, sp2, *args, **kwargs):
    mu_list = list(map(float, sp1))
    lambda_list = list(map(float, sp2))

    x = np.linspace(-10, 10, 1000)
    labels = []
    plt.figure(dpi=150)
    for mu, la in zip(mu_list, lambda_list):
        labels.append('μ={}, λ={}'.format(mu, la))
        y = laplace.pdf(x, loc=mu, scale=la)
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Laplace distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()


def T_Distribution(sp1, *args, **kwargs):
    df_list = list(map(float, sp1))

    x = np.linspace(-5, 5, 1000)
    labels = []
    plt.figure(dpi=150)
    for df in df_list:
        labels.append('df={}'.format(df))
        y = t.pdf(x, df=df)
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('T distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()


def Expon_Distribution(sp1, *args, **kwargs):
    lambda_list = list(map(float, sp1))

    x = np.linspace(0, 5, 100)
    labels = []
    plt.figure(dpi=150)
    for la in lambda_list:
        labels.append('λ={}'.format(la))
        y = expon.pdf(x, scale=1 / la)
        plt.axis([0, 5, 0, 1.5])
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.legend(labels=labels, loc='best')
    plt.title('Exponential distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()


def Zipf_Distribution(sp1, *args, **kwargs):
    a_list = list(map(float, sp1))

    x = np.arange(1, 11)
    labels = []
    plt.figure(dpi=150)
    ax = plt.gca(xscale='log', yscale='log')
    for a in a_list:
        labels.append('a={}'.format(a))
        y = zipf.pmf(k=x, a=a)
        plt.xticks(np.arange(1, 11), np.arange(1, 11))
        plt.tick_params(axis='both', labelsize=10)
        plt.scatter(x, y)
        plt.plot(x, y)
    plt.legend(labels=labels, loc='best')
    plt.title('Zipf distribution')
    plt.show()


def Weibull_Distribution(sp1, sp2, *args, **kwargs):
    lambda_list = list(map(float, sp1))
    a_list = list(map(float, sp2))

    x = np.linspace(0, 2.5, 1000)
    labels = []
    plt.figure(dpi=150)
    for la, a in zip(lambda_list, a_list):
        labels.append('λ={}, a={}'.format(la, a))
        y = weibull_min.pdf(x, c=a, scale=la)
        plt.tick_params(axis='both', labelsize=10)
        plt.axis([0, 2.5, 0, 2.5])
        plt.plot(x, y)
    plt.legend(labels=labels, loc='best')
    plt.title('Weibull distribution')
    plt.show()


def Lomax_Distribution(sp1, sp2, *args, **kwargs):
    lambda_list = list(map(float, sp1))
    alpha_list = list(map(float, sp2))

    x = np.linspace(0, 6, 1000)
    labels = []
    plt.figure(dpi=150)
    for la, alpha in zip(lambda_list, alpha_list):
        labels.append('λ={}, α={}'.format(la, alpha))
        y = lomax.pdf(x, c=alpha, scale=la)
        # plt.axis([0, 6, 0, 2])
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Lomax distribution')
    plt.show()


def Negative_Binomial_Distribution(sp1, sp2, *args, **kwargs):
    N_list = list(map(float, sp1))
    p_list = list(map(float, sp2))

    x = np.arange(50)
    labels = []
    plt.figure(dpi=150)
    for p, n in zip(p_list, N_list):
        labels.append('p={}, n={}'.format(p, n))
        y = nbinom.pmf(x, n=n, p=p)
        plt.tick_params(axis='both', labelsize=10)
        plt.scatter(x, y)
        plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Negative binomial distribution')
    plt.xlabel('N')
    plt.ylabel('Frequency')
    plt.show()


def Logistic_Distribution(sp1, sp2, *args, **kwargs):
    mu_list = list(map(float, sp1))
    s_list = list(map(float, sp2))

    x = np.linspace(-5, 25, 1000)
    labels = []
    plt.figure(dpi=150)
    for mu, s in zip(mu_list, s_list):
        labels.append('μ={}, s={}'.format(mu, s))
        y = logistic.pdf(x, loc=mu, scale=s)
        plt.tick_params(axis='both', labelsize=10)
        plt.plot(x, y)
    plt.ylim(ymin=0)
    plt.legend(labels=labels, loc='best')
    plt.title('Logistic distribution')
    plt.xlabel('N')
    plt.ylabel('Frequency')
    plt.show()
