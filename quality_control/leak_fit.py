from __future__ import division
import numpy as np
import sys
from scipy import stats
import pints

class LeakModel(pints.ForwardModel):
    def __init__(self, voltage_trace):
        self._V = voltage_trace

    def simulate(self, parameters, times):
        g, E = parameters
        return g * (self._V[:len(times)] - E)

    def dimension(self):
        return 2


def linear_and_diode(a, v, T):
    # a[0] = g_leak
    # a[1] = E_leak
    # a[2] = Reverse bias saturation current Is = Is(T), typically ~pA
    # a[3] = ideality factor n, typically 1 < n < 2
    k = 1.38064852e-23
    q = 1.6021766208e-19
    kTq = k * T / q * 1000 # mV

    if a[0] < 0:
        return v * float('inf')
    if a[2] < 0:
        return v * float('inf')
    if a[3] <= 0:  # 0 is not valid too
        return v * float('inf')
    return a[0] * (v - a[1]) + a[2] * (np.exp(v / (a[3] * kTq)) - 1)


def nonlinearleak_error(a, i, v, T):
    return np.mean((linear_and_diode(a, v, T) - i) ** 2)


def fft(current, voltage, title, show=False, cell=None):
    import scipy.fftpack

    n = len(current)
    t = 0.2 * 1e-3
    yf = scipy.fftpack.fft(current)
    yf = 2 / n * np.abs(yf[0:n // 2])
    xf = np.linspace(0, 1 / (2 * t), n // 2)

    i = np.where(xf < 100)[0][-1]
    maxfft = np.max(yf[i:])
    print('Max FFT above ' + str(xf[i]) + ' Hz')
    print(maxfft)

    if show:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure for report
        plt.figure(figsize=(4,2))
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.plot(xf, yf, label='Cell ' + str(cell))
        plt.legend()
        plt.ylim(0, 0.5)
        plt.xlim(0, 200)
        plt.tight_layout()
        plt.savefig('fft-cell-' + str(cell))

        # Figure for viewing
        plt.figure()
        plt.suptitle(title)

        plt.subplot(3, 1, 1)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.plot(xf, yf, label='Current')
        plt.legend()
        plt.ylim(0, 1.3 * np.max(yf[1000:]))

        plt.subplot(3, 1, 2)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.plot(xf, yf, label='Current')
        f1 = 7 / (2 * np.pi)
        f2 = 37 / (2 * np.pi)
        f3 = 190 / (2 * np.pi)
        plt.axvline(f1 , color='C2', label=str(np.round(f1, 2)) + ' Hz')
        plt.axvline(f2, color='C3', label=str(np.round(f2, 2)) + ' Hz')
        plt.axvline(f3, color='C4', label=str(np.round(f3, 1)) + ' Hz')
        plt.xlim(0, 60)
        plt.legend()

        yf = scipy.fftpack.fft(voltage)
        yf = 2 / n * np.abs(yf[0:n // 2])
        plt.subplot(3, 1, 3)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.plot(xf, yf, label='Voltage')
        plt.axvline(f1 , color='C2', label=str(np.round(f1, 2)) + ' Hz')
        plt.axvline(f2, color='C3', label=str(np.round(f2, 2)) + ' Hz')
        plt.axvline(f3, color='C4', label=str(np.round(f3, 1)) + ' Hz')
        plt.xlim(0, 60)
        plt.ylim(0, 50)
        plt.legend()

    return maxfft


def fit(model, values, times, N=3):
    sigma_noise = np.std(values[:200])
    problem = pints.SingleSeriesProblem(model, times, values)
    log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
    boundaries = pints.Boundaries(
            [0, -120],
            [10, 100])

    params, scores = [], []
    for i in range(N):
        # Randomly pick a starting point
        x0 = boundaries.sample()[0]
        # x0 = np.copy(transform_trueparams)
        print('Starting point: ', x0)

        # Create optimiser
        print('Starting log_likelihood: ', log_likelihood(x0))
        opt = pints.Optimisation(log_likelihood, x0.T, boundaries=boundaries,
                                 method=pints.XNES)
        opt.set_max_iterations(None)
        opt.set_parallel(True)
        opt.set_log_to_screen(False)
        # opt.optimiser().set_tolfun(1e-1)  # probably enough...

        # Run optimisation
        try:
            with np.errstate(all='ignore'): # Tell numpy not to issue warnings
                p, s = opt.run()
                params.append(p)
                scores.append(s)
                print('Found solution:')
                for k, x in enumerate(p):
                    print(pints.strfloat(x))
        except ValueError:
            import traceback
            traceback.print_exc()

    # Order from best to worst
    order = np.argsort(scores)[::-1]  # (use [::-1] for LL)
    scores = np.asarray(scores)[order]
    params = np.asarray(params)[order]
    # Show results
    bestn = min(3, N)
    print('Best %d log_likelihoods:' % bestn)
    for i in xrange(bestn):
        print(scores[i])
    print('Mean & std of log_likelihood:')
    print(np.mean(scores))
    print(np.std(scores))
    print('Worst log_likelihood:')
    print(scores[-1])

    # Extract best
    return params[0], scores[0]


def fit_leak_lr(staircase_protocol, current, percentage_to_remove=0, V_full=[-120, -80],
                ramp_start=300, ramp_end=700, dt=1e-1, extra_points=[]):

    # Fitting leak during the first ramp in staircaseramp prt
    #
    # staircase_protocol: full staircase ramp protocol
    # current: corresponding current for the staircase ramp protocol
    # V_win: Voltage window for fitting (in the direction of time)
    # V_full: Full voltage range during the ramp (in the direction of time)
    # ramp_start: starting time of the ramp that matches the input protocol
    # ramp_end: ending time of the ramp that matches the input protocol
    # dt: duration of each time step to work out the index in the input protocol
    ramp_start += (ramp_end-ramp_start)*(0.01*percentage_to_remove)
    # ramp_end  -= (ramp_end-ramp_start)*(0.005*percentage_to_remove)
    rampi, rampf = int(ramp_start / dt), int(ramp_end / dt)

    n_samples = rampf - rampi
    # idxi = int(np.abs(np.float(V_win[0] - V_full[0]))\
    #         / np.abs(np.float(V_full[1] - V_full[0]))\
    #         * n_samples)
    # idxf = int(np.abs(np.float(V_win[1] - V_full[0]))\
    #         / np.abs(np.float(V_full[1] - V_full[0]))\
    #         * n_samples)
    # idxi = 0
    # idxf = -1
    # Assumed V_win, V_full where given correctly!!
    x = staircase_protocol[rampi:rampf]
    y = current[rampi:rampf]
    x = np.append(x, staircase_protocol[rampi:rampf])
    y = np.append(y, current[rampi:rampf])

    if len(extra_points) > 0:
        extra_voltage = staircase_protocol[extra_points]
        extra_current = current[extra_points]
        x = np.append(x, extra_voltage)
        y = np.append(y, extra_current)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    g_leak = slope
    E_leak = -1 * intercept / slope

    predictions = (x - E_leak) * g_leak
    n = len(x)

    eps2 = (predictions - y)**2

    s_beta = np.sqrt((1/(n-2) * eps2.sum())/((x - x.mean())**2).sum())
    s_alpha = s_beta * np.sqrt((x**2).sum()/n)

    return g_leak, E_leak, r_value, s_alpha, s_beta, x, y



def plot_ramp(cell, before, after, leak_model, param, param2,
        staircase_protocol, temperature, saveas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot leak
    plt.figure(figsize=(10, 14))
    plt.subplot(4, 1, 1)

    plt.axhline(0, c='#7f7f7f')
    plt.plot(before[cell][0], label='Before E-4031')
    leak_before = leak_model.simulate(param, range(len(before[cell][0])))
    plt.plot(leak_before,
             label='g=%s,E=%s' % (param[0], param[1]))
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)

    plt.subplot(4, 1, 2)
    plt.axhline(0, c='#7f7f7f')
    plt.plot(after[cell][0], label='After E-4031')
    leak_after = leak_model.simulate(param2, range(len(after[cell][0])))
    plt.plot(leak_after,
             label='g=%s,E=%s' % (param2[0], param2[1]))
    plt.xlabel('Time index (sample)')
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)

    plt.subplot(4, 1, 3)
    plt.axhline(0, c='#7f7f7f')
    plt.plot(before[cell][0] - leak_before, label='Before E-4031 leak-corrected')
    plt.plot(after[cell][0] - leak_after, label='After E-4031 leak-corrected')
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)

    plt.subplot(4, 1, 4)
    plt.axhline(0, c='#7f7f7f')
    subtracted_all = (before[cell][0] - leak_before) - (after[cell][0] - leak_after)
    plt.plot(subtracted_all,
              label='E-4031-subtracted leak-corrected')
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)
    plt.savefig('%s/%s.png' % (saveas, cell))
    plt.close()


def plot_ramp_2(cell, before, after, leak_before, leak_after, param, param2,
                saveas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot leak
    plt.figure(figsize=(10, 14))
    plt.subplot(4, 1, 1)

    plt.axhline(0, c='#7f7f7f')
    plt.plot(before[0], label='Before E-4031')
    plt.plot(leak_before,
             label='g=%s,E=%s' % (param[0], param[1]))
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)

    plt.subplot(4, 1, 2)
    plt.axhline(0, c='#7f7f7f')
    plt.plot(after[0], label='After E-4031')
    plt.plot(leak_after,
             label='g=%s,E=%s' % (param2[0], param2[1]))
    plt.xlabel('Time index (sample)')
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)

    plt.subplot(4, 1, 3)
    plt.axhline(0, c='#7f7f7f')
    plt.plot(before[0] - leak_before, label='Before E-4031 leak-corrected')
    plt.plot(after[0] - leak_after, label='After E-4031 leak-corrected')
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)

    plt.subplot(4, 1, 4)
    plt.axhline(0, c='#7f7f7f')
    subtracted_all = (before[0] - leak_before) - (after[0] - leak_after)
    plt.plot(subtracted_all,
              label='E-4031-subtracted leak-corrected')
    plt.ylabel('Current [pA]')
    plt.legend(loc=4)
    plt.savefig('%s/%s.png' % (saveas, cell))
    plt.close()


def fit_EK(staircase_protocol, current, V_win=[-80, -96], V_full=[-70, -110],
        ramp_start=14.41, ramp_end=14.51, dt=2e-4):
    # Fitting EK during last ramp in staircaseramp prt
    #
    # staircase_protocol: full staircase ramp protocol
    # current: corresponding current for the staircase ramp protocol
    # V_win: Voltage window for fitting (in the direction of time)
    # V_full: Full voltage range during the ramp (in the direction of time)
    # ramp_start: starting time of the ramp that matches the input protocol
    # ramp_end: ending time of the ramp that matches the input protocol
    # dt: duration of each time step to work out the index in the input protocol
    #
    # Note:
    # V_win=[-80, -96] works quite nicely for temperature at ~25oC
    # V_win=[-70, -80] works quite nicely for temperature at ~37oC
    from scipy import stats
    rampi, rampf = int(ramp_start / dt), int(ramp_end / dt)
    n_samples = rampf - rampi
    idxi = int(np.abs(np.float(V_win[0] - V_full[0]))\
            / np.abs(np.float(V_full[1] - V_full[0]))\
            * n_samples)
    idxf = int(np.abs(np.float(V_win[1] - V_full[0]))\
            / np.abs(np.float(V_full[1] - V_full[0]))\
            * n_samples)
    # Assumed V_win, V_full where given correctly!!
    x = staircase_protocol[rampi:rampf][idxi:idxf]
    y = current[rampi:rampf][idxi:idxf]
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, x)
    print(intercept)
    return intercept


def fit_EK_poly(staircase_protocol, current, deg=3, V_full=[-70, -110],
        ramp_start=14.41, ramp_end=14.51, dt=2e-4, savefig=None, beforeE4031=None):
    # Fitting EK during last ramp in staircaseramp prt
    #
    # staircase_protocol: full staircase ramp protocol
    # current: corresponding current for the staircase ramp protocol
    # deg: n degree of polynomial for fitting
    # V_full: Full voltage range during the ramp (in the direction of time)
    # ramp_start: starting time of the ramp that matches the input protocol
    # ramp_end: ending time of the ramp that matches the input protocol
    # dt: duration of each time step to work out the index in the input protocol
    # savefig: for debug use, ['save_name', temperature]
    #
    # Note:
    # V_win=[-80, -96] works quite nicely for temperature at ~25oC
    # V_win=[-70, -80] works quite nicely for temperature at ~37oC
    rampi, rampf = int(ramp_start / dt), int(ramp_end / dt)
    assert((rampf - rampi) > deg + 1)
    vmin, vmax = np.min(V_full), np.max(V_full)
    x = staircase_protocol[rampi:rampf]
    y = current[rampi:rampf]
    p = np.poly1d(np.polyfit(x, y, deg))
    if savefig is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label='Data')
        plt.plot(x, p(x), label='Fitted')
        if beforeE4031 is not None:
            plt.plot(x, beforeE4031[rampi:rampf], label='Before E4031')
        temperature = savefig[1]  # K
        const_R = 8.314472  # J/mol/K
        const_F = 9.64853415e4  # C/mol
        const_Ko = 4.0  # mM (my hERG experiments)
        const_Ki = 110.0  # mM (my hERG experiments)
        RTF = const_R * temperature / const_F  # J/C == V
        EK = RTF * np.log(const_Ko / const_Ki) * 1000  # mV
        plt.axvline(EK, ls='--', c='#7f7f7f', label=r'Expected $E_K$')
        plt.axhline(0, c='#7f7f7f')
        plt.xlabel('Voltage [mV]')
        plt.ylabel('Current [pA]')
        plt.legend(loc=4)
        plt.savefig(savefig[0])
        plt.close()
    # check within range V_full
    r = []
    for i in p.r:
        if vmin < i <= vmax and (np.isreal(i) or np.abs(i.imag) < 1e-8):
            r.append(i)
    print('Found EK: ', r)
    if len(r) == 1:
        return r[0].real
    elif len(r) > 1:
        return np.max(r).real
    else:
        return np.inf
