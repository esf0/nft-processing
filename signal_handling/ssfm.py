import numpy as np
# import multiprocessing

from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
from numba import jit, njit

# from numba import jit
# import time

# start = time.time()


def ssfm_dispersive_step(signal, t_span, dispersion=None, w=None, delta_z=0.001, alpha=0, beta2=1, beta3=0):
    # F+
    temp_freq = fft(signal)

    if dispersion is None:
        if w is None:
            # w is frequencies in Fourier-space for Split-step method
            # w is defined as w = -W/2 : dw : W/2
            # W is 1 / dt, where dt is initial signal time step
            # dw = W / N, where N is number of point in initial signal

            n = len(signal)
            # dw = band / n
            # w = [dw * (i - n / 2) for i in range(n)]
            w = fftshift([(i - n / 2) * (2. * np.pi / t_span) for i in range(n)])
            # w = np.array([(i - n / 2) * (2. * np.pi / t_span) for i in range(n)])
            # w = np.fft.fftfreq(K, d=t_span/n) * 2. * np.pi # Probably better way

        dispersion = np.exp((0.5j * beta2 * w ** 2 + 1. / 6. * beta3 * w ** 3 - alpha / 2.) * delta_z)

    # F-
    # print(np.mean(dispersion))
    temp_signal = ifft(temp_freq * dispersion)

    return temp_signal

@njit
def ssfm_nonlinear_step(signal, gamma, delta_z):
    temp_signal = signal * np.exp(1.0j * delta_z * gamma * np.power(np.absolute(signal), 2))

    return temp_signal


def fiber_propagate(initial_signal, t_span, fiber_length, n_span, gamma, beta2, alpha=0, beta3=0):

    if abs(fiber_length) < 1e-15:
        return initial_signal

    dz = fiber_length / n_span
    signal_length = len(initial_signal)

    w = fftshift([(i - signal_length / 2) * (2. * np.pi / t_span) for i in range(signal_length)])

    dispersion = np.exp((0.5j * beta2 * w ** 2 + 1. / 6. * beta3 * w ** 3 - alpha / 2.) * dz)
    dispersion_half_p = np.exp((0.5j * beta2 * w ** 2 + 1. / 6. * beta3 * w ** 3 - alpha / 2.) * dz * 0.5)
    dispersion_half_m = np.exp((0.5j * beta2 * w ** 2 + 1. / 6. * beta3 * w ** 3 - alpha / 2.) * (-dz) * 0.5)

    # D/2
    signal = ssfm_dispersive_step(initial_signal, t_span, dispersion=dispersion_half_p, delta_z=dz / 2., beta2=beta2,
                                  alpha=alpha, beta3=beta3)

    for n in range(n_span):
        signal = ssfm_nonlinear_step(signal, gamma, dz)
        signal = ssfm_dispersive_step(signal, t_span, dispersion=dispersion, delta_z=dz, beta2=beta2, alpha=alpha,
                                      beta3=beta3)

    # -D/2
    signal = ssfm_dispersive_step(signal, t_span, dispersion=dispersion_half_m, delta_z=-dz / 2., beta2=beta2,
                                  alpha=alpha, beta3=beta3)

    return signal


def fiber_propagate_high_order(initial_signal, t_span, fiber_length, n_span, gamma, beta2, alpha=0, beta3=0):
    # TODO: check dz and n_span for calculation

    dz = fiber_length / (6 * n_span)

    signal = initial_signal
    # One step gives z + 6dz
    for step in range(n_span):
        # (D/2 N)^4
        for n in range(4):
            signal = ssfm_dispersive_step(signal, t_span, delta_z=dz / 2., beta2=beta2)
            signal = ssfm_nonlinear_step(signal, gamma, dz)

        # -D/2
        signal = ssfm_dispersive_step(signal, t_span, delta_z=-dz / 2., beta2=beta2)

        # -2N
        signal = ssfm_nonlinear_step(signal, gamma, -2.0 * dz)

        # -D/2
        signal = ssfm_dispersive_step(signal, t_span, delta_z=-dz / 2., beta2=beta2)

        # N
        signal = ssfm_nonlinear_step(signal, gamma, dz)

        # (D/2 N)^3
        for n in range(3):
            signal = ssfm_dispersive_step(signal, t_span, delta_z=dz / 2., beta2=beta2)
            signal = ssfm_nonlinear_step(signal, gamma, dz)

        # D/2
        signal = ssfm_dispersive_step(signal, t_span, delta_z=dz / 2., beta2=beta2)

    return signal


def check_energy(signal, t_span, spectrum):
    # energy_signal = np.mean(np.power(np.absolute(signal), 2)) * t_span
    # energy_spectrum = np.mean(np.power(np.absolute(spectrum), 2)) * (2.0 * np.pi / t_span * len(signal))
    energy_signal = np.mean(np.power(np.absolute(signal), 2))
    energy_spectrum = np.mean(np.power(np.absolute(spectrum), 2)) / len(signal)
    if abs(energy_signal - energy_spectrum) > 1e-14:
        print("Error, energy is different: ", abs(energy_signal - energy_spectrum))

    return energy_signal, energy_spectrum


def get_energy(signal, t_span):
    return np.mean(np.power(np.absolute(signal), 2)) * t_span


def get_gauss_pulse(amplitude, t, tau, z=0, beta2=0):
    z_ld = z / tau ** 2 * abs(beta2)
    a_z = amplitude / np.sqrt(1 - 1.0j * z_ld * np.sign(beta2))

    return a_z * np.exp(-0.5 / (1 + z_ld ** 2) * np.power(t / tau, 2) * (1.0 + 1.0j * z_ld))

    # return amplitude * np.exp(-0.5 * np.power(t / tau, 2))


def get_pulse_nonlinear(signal, gamma, z):
    return signal * np.exp(1.0j * gamma * np.power(np.abs(signal), 2) * z)


def get_soliton_pulse(t, tau, soliton_order, beta2, gamma):
    if beta2 > 0:
        print("Error: beta2 > 0")
        beta2 = -beta2
    return np.sqrt(-beta2 * soliton_order ** 2 / (gamma * tau ** 2)) / np.cosh(t / tau)