"""
Distribution fitting utilities for power-law (f(n)) and exponential (h(n)) forms.

Both pow_calc and exp_calc return a 7-tuple:
    fx, fy, coeff, coeff_se, inter, inter_se, indic

where coeff_se / inter_se are standard errors from scipy.stats.linregress.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

IQRdiv   = 2
s_time   = 100000
bincross = 10
delthres = 1e-8


def get_distr(xs):
    distr = {}
    for x in xs:
        distr[x] = distr.setdefault(x, 0) + 1
    return distr


def get_logbin_mix(distr, binsize, binstart, bincross):
    distr_bin = {}
    bin_mix   = {}
    b0 = binstart
    b1 = b0 + binsize
    bc = b0 + binsize * 0.5

    for x in sorted(distr.keys()):
        while math.log(x) >= b1:
            b0 += binsize
            b1  = b0 + binsize
            bc  = b0 + binsize * 0.5
        xc = math.exp(bc)
        distr_bin[xc]  = distr_bin.setdefault(xc, 0) + distr[x]
        if x < bincross:
            bin_mix[xc] = bin_mix.setdefault(xc, 0) + 1
        else:
            bin_mix[xc] = math.exp(b1) - math.exp(b0)

    return distr_bin, bin_mix


def pow_calc(real):
    """
    Fit a power-law to `real` via log-log linear regression on log-binned histogram.

    Returns
    -------
    fx, fy        : list  binned x / normalized y values used for fitting
    coeff         : float power-law exponent (positive; slope = -coeff)
    coeff_se      : float standard error of coeff
    inter         : float prefactor (exp of intercept)
    inter_se      : float standard error of inter (propagated)
    indic         : int   1 if fit was performed, 0 otherwise
    """
    if len(real) <= 1:
        return [], [], 0, 0, 0, 0, 0

    real = np.array(real)
    q25, q75 = np.log(np.quantile(real, 0.25)), np.log(np.quantile(real, 0.75))
    IQR      = q75 - q25
    binsize  = 2 * IQR / (len(real) ** (1 / 3)) / IQRdiv
    binstart = math.log(min(real))

    distr_bin, bin_mix = get_logbin_mix(get_distr(real), binsize, binstart, bincross)
    for x in distr_bin:
        distr_bin[x] /= bin_mix[x]

    xs = sorted(distr_bin.keys())
    ys = np.array([distr_bin[i] for i in xs])
    ys = ys / ys.sum()

    fx, fy = [], []
    for xi, yi in zip(xs, ys):
        if yi > delthres and xi > delthres:
            fx.append(xi - xs[0] + 1)
            fy.append(yi)

    if len(fx) <= 1:
        return fx, fy, 0, 0, 0, 0, 0

    logx   = np.log(np.array(fx))
    logy   = np.log(np.array(fy))
    result = linregress(logx, logy)

    coeff    = -result.slope
    coeff_se = result.stderr
    inter    = math.exp(result.intercept)
    # propagate uncertainty: se(exp(b)) ≈ exp(b) * se(b)
    inter_se = inter * result.intercept_stderr

    return fx, fy, coeff, coeff_se, inter, inter_se, 1


def exp_calc(real):
    """
    Fit an exponential decay to `real` via linear regression on log(y) vs x.

    Returns
    -------
    fx, fy        : list  binned x / normalized y values used for fitting
    coeff         : float decay rate (positive; slope = -coeff)
    coeff_se      : float standard error of coeff
    inter         : float prefactor (exp of intercept)
    inter_se      : float standard error of inter (propagated)
    indic         : int   1 if fit was performed, 0 otherwise
    """
    if len(real) <= 1:
        return [], [], 0, 0, 0, 0, 0

    real = np.array(real)
    q25, q75 = np.quantile(real, 0.25), np.quantile(real, 0.75)
    IQR      = q75 - q25
    binsize  = 2 * IQR / (len(real) ** (1 / 3)) / IQRdiv
    binstart = 0 if min(real) == 1 else min(real)
    bins     = min(int(s_time / binsize), 10000)

    int_dis, x, _ = plt.hist(real, bins=bins, density=True)
    plt.clf(); plt.cla(); plt.close()
    x  = 0.5 * (x[1:] + x[:-1])
    y  = int_dis / int_dis.sum()

    fx, fy = [], []
    for xi, yi in zip(x, y):
        if yi > delthres and xi > delthres:
            fx.append(xi - x[0] + 1)
            fy.append(yi)

    if len(fx) <= 1:
        return fx, fy, 0, 0, 0, 0, 0

    logx   = np.array(fx)
    logy   = np.log(np.array(fy))
    result = linregress(logx, logy)

    coeff    = -result.slope
    coeff_se = result.stderr
    inter    = math.exp(result.intercept)
    inter_se = inter * result.intercept_stderr

    return fx, fy, coeff, coeff_se, inter, inter_se, 1


def h_calc(real, bins=100):
    """
    Fit an exponential decay to state-duration data h(n) using fixed-width bins.

    Uses fixed `bins` count (default 100) without x-shifting, matching the
    approach in fitting.py that produced the paper's Table 2 α values.

    Returns
    -------
    fx, fy        : list  bin-center x / normalized y values used for fitting
    coeff         : float decay rate α (positive; slope = -α)
    coeff_se      : float standard error of α
    inter         : float prefactor (exp of intercept)
    inter_se      : float standard error of inter (propagated)
    indic         : int   1 if fit was performed, 0 otherwise
    """
    if len(real) <= 1:
        return [], [], 0, 0, 0, 0, 0

    real = np.array(real)
    int_dis, x, _ = plt.hist(real, bins=bins, density=True)
    plt.clf(); plt.cla(); plt.close()
    x = 0.5 * (x[1:] + x[:-1])
    y = int_dis / int_dis.sum() if int_dis.sum() != 0 else int_dis

    fx, fy = [], []
    for xi, yi in zip(x, y):
        if yi > delthres and xi > delthres:
            fx.append(xi)
            fy.append(yi)

    if len(fx) <= 1:
        return fx, fy, 0, 0, 0, 0, 0

    logy   = np.log(np.array(fy))
    result = linregress(np.array(fx), logy)

    coeff    = -result.slope
    coeff_se = result.stderr
    inter    = math.exp(result.intercept)
    inter_se = inter * result.intercept_stderr

    return fx, fy, coeff, coeff_se, inter, inter_se, 1
