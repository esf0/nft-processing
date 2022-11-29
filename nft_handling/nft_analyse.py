import sys

# adding signal_handling to the system path
sys.path.insert(0, '../signal_handling/')
sys.path.insert(0, '../pjt/')

from importlib import reload
import FNFTpy

reload(FNFTpy)
from FNFTpy import nsev, nsev_poly
from FNFTpy import nsev_inverse, nsev_inverse_xi_wrapper
import numpy as np
import scipy as sp
from scipy.integrate import simps, trapz
from scipy.special import gamma
from scipy.linalg import companion, eigvals
from dataclasses import dataclass
import pandas as pd
import random
import timeit
import time
from tqdm import tqdm
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from ssfm import fiber_propagate, get_soliton_pulse, get_gauss_pulse, get_energy
import matplotlib.pyplot as plt
import matplotlib

import signal_generation as sg
reload(sg)

import test_signals
from PJTpy import pjt

import warnings
from datetime import datetime
from numba import jit, njit

# from prettytable import PrettyTable


def print_calc_time(start_time, type_of_calc=''):
    end_time = datetime.now()
    total_time = end_time - start_time
    print('Time to calculate ' + type_of_calc, total_time.total_seconds() * 1000, 'ms')


def get_raised_cos(t_span, n):
    dt = t_span / (n - 1)

    f = np.zeros(n)
    for k in range(n):
        t = dt * k - t_span / 2
        if t < -t_span / 4 or t > t_span / 4:
            f[k] = 0.5 * (1 - np.cos(4 * np.pi * t / t_span + 2 * np.pi))
        else:
            f[k] = 1.0

    return f


def get_raised_contour(ampl, xi_span, n_xi):
    """
    Return raised contour in complex plane
    Args:
        ampl: amplitude of imaginary part
        xi_span: full spectral interval
        n_xi: number of points

    Returns:
        Array of points

    """
    raised = get_raised_cos(xi_span, n_xi)
    d_xi = xi_span / (n_xi - 1)
    xi = np.array([-xi_span / 2 + i * d_xi for i in range(n_xi)]) + 1j * ampl * raised

    return xi


def is_arg_jump(first, second):
    if ((first.imag > 0) & (second.imag < 0) &
        (first.real < 0) & (second.real < 0)) | ((first.imag < 0) & (second.imag > 0) &
                                                 (first.real < 0) & (second.real < 0)):
        return 1
    return 0


def get_rect(start, end, n_horizontal, n_vertical, n_top=-1):
    if n_top == -1:
        n_top = n_horizontal
    step_v = (end.imag - start.imag) / (n_vertical - 1)
    step_h = (end.real - start.real) / (n_horizontal - 1)
    step_t = (end.real - start.real) / (n_top - 1)

    xi = np.array([start + step_h * i for i in range(n_horizontal)] +
                  [end.real + 1.0j * start.imag + 1.0j * step_v * i for i in range(1, n_vertical)] +
                  [end - step_t * i for i in range(1, n_top)] +
                  [start.real + 1.0j * end.imag - 1.0j * step_v * i for i in range(1, n_vertical - 1)], dtype=complex)

    return xi


def get_rect_filled(start, end, n_horizontal, n_vertical):
    step_v = (end.imag - start.imag) / (n_vertical - 1)
    step_h = (end.real - start.real) / (n_horizontal - 1)
    xi = np.zeros(n_vertical * n_horizontal, dtype=np.complex128)
    for i in range(n_vertical):
        for k in range(n_horizontal):
            xi[i * n_horizontal + k] = complex(start.real + step_h * k, start.imag + step_v * i)
    return xi


def get_phase_shift(first, second, result_type=0):
    error = 0
    delta_phi = 0

    if (first.imag >= 0) & (second.imag < 0) & (first.real <= 0) & (second.real <= 0):
        delta_phi = np.angle(second) - np.angle(first) + 2. * np.pi
        error = 1
    elif (first.imag < 0) & (second.imag >= 0) & (first.real <= 0) & (second.real <= 0):
        delta_phi = np.angle(second) - np.angle(first) - 2. * np.pi
        error = -1
    elif (first.imag * second.imag < 0) & (first.real * second.real < 0):
        if result_type == 0:
            error = -10.  # I chose -10 cause phase shift can not be more than 2pi
        elif result_type == 1:
            pass
            # TODO: minimal distance
    else:
        delta_phi = np.angle(second) - np.angle(first)

    return delta_phi, error


def get_cauchy(order, a, z, func_values):
    value = np.math.factorial(order) / (2.0j * np.pi) * simps(func_values / np.power(z - a, order + 1), z)
    return value


def get_a_cauchy(point, xi_cont, a_xi):
    return get_cauchy(0, point, xi_cont, a_xi - 1) + 1


def get_contour_phase_shift_adaptive(q, t, xi_cont, a_xi):
    total_phase_shift = 0
    error_code = 0
    jump = []

    for i in range(len(a_xi) - 1):
        d_phi, error_phi = get_phase_shift(a_xi[i], a_xi[i + 1])
        if error_phi == -10:
            xi_sub = np.array([xi_cont[i], (xi_cont[i] + xi_cont[i + 1]) / 2., xi_cont[i + 1]])
            # here calc a_xi_sub
            res_sub = nsev(q, t, M=3, Xi1=xi_cont[i], Xi2=xi_cont[i + 1], kappa=1, cst=1, dst=3)
            a_xi_sub = res_sub['cont_a']

            current_phase_shift, error_code_sub, jump_sub = get_contour_phase_shift_adaptive(q, t, xi_sub, a_xi_sub)
            for one_jump in jump_sub:
                jump.append(one_jump)

            if error_code_sub == -1:
                print("[get_contour_phase_shift_adaptive] Error: problem in algorithm")
                error_code = error_code_sub

            d_phi = current_phase_shift
        elif (error_phi == 1) or (error_phi == -1):
            one_jump = {"xi": xi_cont[i], "xi_next": xi_cont[i + 1], "sign": error_phi}
            jump.append(one_jump)

        total_phase_shift += d_phi

    return total_phase_shift, error_code, jump


def make_dtib(q, t, sign):
    h = t[1] - t[0]

    n_omega = len(q)
    omega = np.zeros(n_omega)

    n_beta = n_omega
    beta = np.zeros(n_beta)
    c_coef = np.zeros(n_beta)
    d_coef = np.zeros(n_beta)

    y = np.zeros(())

    omega[0] = -0.5 * q[0]
    # m = 1
    for m in range(1, n_omega):
        beta[m] = -0.5 * h * q[m]
        sum = 0
        for j in range(1, m):
            sum += omega[m - j] * y[j][m]
        omega[m] = 1. / (h * y[0][m]) * (beta[m] - h * sum)

        c_coef[m] = 1. / (1 + sign * np.power(np.absolute(beta[m]), 2))
        d_coef[m] = -beta[m] * c_coef[m]

    # TODO: finish code


@njit
def swap(a, b):
    temp = a
    a = b
    b = temp

    return a, b


@njit
def make_itib(omega, t, split_index=-1, sigma=1):
    """
    Inverse TIB algorithm to restore signal q form kernel

    Args:
        omega: Kernel function
        t: time grid
        sigma: defines focusing (+1) and defocusing (-1) cases

    Returns:
        q: restored signal in time points

    """
    # sigma = 1 focusing case
    # sigma = -1 defocusing case

    sigma = -1  # focusing / defocusing case = -1 / +1

    dt = t[1] - t[0]
    inv_dt = -2.0 / dt
    n = len(t)
    q = np.zeros(n, dtype=np.complex128)

    q[0] = -2 * omega[0]

    if split_index == -1:
        split_index = n

    r = omega.copy()

    r *= dt

    r[0] = r[0] / 2

    y = np.zeros(n + 1, dtype=np.complex128)
    y0 = np.zeros(n + 1, dtype=np.complex128)

    z = np.zeros(n + 1, dtype=np.complex128)
    z0 = np.zeros(n + 1, dtype=np.complex128)

    y[0] = np.array(1.0 / (1 - sigma * r[0] * np.conj(r[0])))

    z[0] = -r[0] * y[0]

    r = r[::-1]

    for m in range(1, split_index):
        bet = np.sum(r[n - m - 1:n] * y[0:m + 1])

        c_m = (1 / (1 - sigma * bet * np.conj(bet)))

        d_m = -bet * c_m

        y, y0 = swap(y, y0)
        z, z0 = swap(z, z0)

        y[0:m + 1] = c_m * y0[0:m + 1] + sigma * d_m * np.conj(z0[m::-1])

        z[0:m + 1] = c_m * z0[0:m + 1] + d_m * np.conj(y0[m::-1])

        q[m] = bet * inv_dt

    return q


def make_itib_egor(omega, t, sigma=1):
    """
    Inverse TIB algorithm to restore signal q form kernel

    Args:
        omega: Kernel function
        t: time grid
        sigma: defines focusing (+1) and defocusing (-1) cases

    Returns:
        q: restored signal in time points

    """
    # sigma = 1 focusing case
    # sigma = -1 defocusing case
    n_t = len(t)
    dt = t[1] - t[0]

    q = np.zeros(n_t, dtype=np.complex128)
    q[0] = -2.0 * omega[0]

    # y = np.zeros(n_t, dtype=np.complex128)
    # z = np.zeros(n_t, dtype=np.complex128)

    y_prev = np.array([1.0 / (1.0 + sigma * dt ** 2 * np.absolute(omega[0]) ** 2 / 4.0)])
    z_prev = np.array([-0.5 * y_prev[0] * dt * omega[0]])

    beta = np.zeros(n_t, dtype=np.complex128)

    for m in range(1, n_t):

        sum = 0
        for j in range(m):
            sum += omega[m - j] * y_prev[j]

        beta[m] = dt * sum
        q[m] = -2.0 * beta[m] / dt

        c = 1.0 / (1 + sigma * np.absolute(beta[m]) ** 2)
        d = -beta[m] * c

        y_next = c * np.append(y_prev, 0) - d * sigma * np.conj(np.append(0, z_prev[::-1]))
        z_next = c * np.append(z_prev, 0) + d * np.conj(np.append(0, y_prev[::-1]))

        y_prev = y_next
        z_prev = z_next

    return q


def make_itib_other(omega, t, sigma=1):
    # sigma = 1 focusing case
    # sigma = -1 defocusing case
    n_t = len(t)
    dt = t[1] - t[0]

    q = np.zeros(n_t, dtype=np.complex128)
    omega = omega * 2.0 * dt

    q[0] = -omega[0]
    y_prev = np.array([1.0 / (1 + abs(omega[0]) ** 2)])
    z_prev = np.array([-y_prev * omega[0]])

    for m in range(1, n_t):
        # print(len(y_prev), len(omega[m - 1::-1]))
        q[m - 1] = np.dot(-y_prev, omega[m - 1::-1])
        c = 1.0 / (1.0 + abs(q[m - 1]) ** 2)
        d = q[m - 1] * c

        y_next = c * np.append(y_prev, 0) - d * sigma * np.conj(np.append(0, z_prev[m - 1::-1]))
        z_next = c * np.append(z_prev, 0) + d * np.conj(np.append(0, y_prev[m - 1::-1]))

        y_prev = y_next
        z_prev = z_next

    q[-1] = np.dot(-y_prev, omega[::-1])
    q = q / dt

    return q


def make_itib_from_scattering(r, xi, rd, xi_d, t, split_index, direction='left', print_sys_message=False):

    coef_t = 2.0

    start_time = datetime.now()
    omega_r = get_omega_continuous(r, xi, coef_t * t, direction)
    if print_sys_message:
        print_calc_time(start_time, 'continuous part of Omega')

    start_time = datetime.now()
    if direction == 'left':
        omega_d = get_omega_discrete(rd, xi_d, coef_t * t)
    else:
        omega_d = get_omega_discrete(rd, xi_d, -coef_t * t)
    if print_sys_message:
        print_calc_time(start_time, 'discrete part of Omega')

    start_time = datetime.now()
    q_tib = make_itib(omega_d + omega_r, coef_t * t, split_index)
    if print_sys_message:
        print_calc_time(start_time, 'TIB')

    return q_tib


def do_bi_direct(q, t, xi, type='orig'):
    """
    Bi-directional algorithm to calculate residues and norming constants for one discrete spectrum point

    Args:
        q: signal
        t: time grid
        xi: spectral parameter where we calculate b(xi) (r(xi))
        type: type of calculation scheme

    Returns:
        b, b / ad, ad

        - b -- b-coefficient (norming constant)
        - r -- r-coefficient (residue)
        - ad -- derivative of a-coefficient in point xi

    """
    # warnings.filterwarnings("error")

    sigma = 1  # focusing case always

    dt = t[1] - t[0]
    n = len(q)
    t_span = t[-1] - t[0]

    x = np.zeros((n + 1, 2), dtype=np.complex128)
    xd = np.zeros((n + 1, 2), dtype=np.complex128)
    y = np.zeros((n + 1, 2), dtype=np.complex128)

    if type == 'orig':
        x[0] = np.array([1, 0])
        xd[0] = np.array([0, 0])
        y[-1] = np.array([0, 1])
    else:
        x[0] = np.array([1, 0]) * np.exp(-1.0j * xi * (-t_span / 2.0 - dt / 2.0))
        xd[0] = np.array([0, 0]) * -1.0j * (-t_span / 2.0 - dt / 2.0) * np.exp(-1.0j * xi * (-t_span / 2.0 - dt / 2.0))
        # y[-1] = np.array([0, 1]) * np.exp(-1.0j * xi * (t_span / 2.0 + dt / 2.0))
        y[-1] = np.array([0, 1]) * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))

    for k in range(n):

        if type == 'orig':
            a_matrix = np.array([[1.0, dt * q[k] * np.exp(2.0j * xi * t[k])],
                                 [-dt * np.conj(q[k]) * np.exp(-2.0j * xi * t[k]), 1.0]])
            ad_matrix = np.array([[0.0, 2.0j * t[k] * dt * q[k] * np.exp(2.0j * xi * t[k])],
                                  [2.0j * t[k] * dt * np.conj(q[k]) * np.exp(-2.0j * xi * t[k]), 0.0]])
            b_matrix = np.array([[1.0, -dt * q[-k - 1] * np.exp(2.0j * xi * t[-k - 1])],
                                 [dt * np.conj(q[-k - 1]) * np.exp(-2.0j * xi * t[-k - 1]), 1.0]])
        else:
            a_matrix = get_transfer_matrix(q, dt, k, xi, type, sigma)
            ad_matrix = get_transfer_matrix(q, dt, k, xi, type + 'd', sigma)
            b_matrix = get_transfer_matrix(q, -dt, -k - 1, xi, type, sigma)

        x[k + 1] = np.matmul(a_matrix, x[k])
        xd[k + 1] = np.matmul(a_matrix, xd[k]) + np.matmul(ad_matrix, x[k])

        y[-k - 2] = np.matmul(b_matrix, y[-k - 1])

    # print(x[0])

    with warnings.catch_warnings(record=True) as w:

        k_target = 0
        diff = 100
        for k in range(n + 1):

            if type == 'orig':
                value = np.absolute(np.absolute(x[k][0]) - 0.5)
            else:
                value = np.absolute(np.absolute(x[k][0] * np.exp(1.0j * xi * (dt * k - t_span / 2.0 - dt / 2.0))) - 0.5)

            if value < diff:
                k_target = k
                diff = value

    # print(k_target)
    # print(x[k_target][0])
    # print(x[-1])
    # print(x[k_target][0] / y[k_target][0])

    if type == 'orig':
        b = x[k_target][0] / y[k_target][0]
        ad = xd[-1][0]
    else:
        # b = x[k_target][0] / y[k_target][0] * np.exp(-1.0j * 2.0 * xi * (t_span / 2.0 + dt / 2.0))
        # b = x[k_target][0] / y[k_target][0] * np.exp(-1.0j * xi * (t_span + dt))  # the closest phase
        b = x[k_target][0] / y[k_target][0]
        ad = xd[-1][0] * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))

    return b, b / ad, ad


def do_bi_direct_array(q, t, xi, type='orig'):
    """
    Bi-directional algorithm to calculate residues and norming constants for array of discrete spectrum points

    Args:
        q: signal
        t: time grid
        xi: array of spectral parameters where we calculate b(xi) (r(xi))
        type: type of calculation scheme

            - 'orig' -- Ablowitz-Ladik scheme
            - 'bo' -- Bofetta-Osborne
            - 'tes4' -- TES4
            - 'es4' -- ES4

    Returns:
        b: array of b-coefficients (norming constants)
        r: array of r-coefficient (residues)
        ad: array of derivative of a-coefficients in points xi

    """
    n_xi = len(xi)
    b = np.zeros(n_xi, dtype=np.complex128)
    r = np.zeros(n_xi, dtype=np.complex128)
    ad = np.zeros(n_xi, dtype=np.complex128)
    for k in range(n_xi):
        b[k], r[k], ad[k] = do_bi_direct(q, t, xi[k], type)

    return b, r, ad


def do_bi_direct_arbitrary(q, t, xi, type='orig'):
    """
        Bi-directional algorithm to calculate b-coefficient for arbitrary spectral point

        Args:
            q: signal
            t: time grid
            xi: spectral parameter where we calculate b(xi) (r(xi))
            type: type of calculation scheme

        Returns:
            b, b / ad, ad

            - b -- b-coefficient (norming constant)
            - r -- r-coefficient (residue)
            - ad -- derivative of a-coefficient in point xi

        """
    # warnings.filterwarnings("error")

    sigma = 1  # focusing case always

    dt = t[1] - t[0]
    n = len(q)
    t_span = t[-1] - t[0]

    x = np.zeros((n + 1, 2), dtype=np.complex128)
    xd = np.zeros((n + 1, 2), dtype=np.complex128)
    y = np.zeros((n + 1, 2), dtype=np.complex128)

    if type == 'orig':
        x[0] = np.array([1, 0])
        xd[0] = np.array([0, 0])
        y[-1] = np.array([0, 1])
    else:
        x[0] = np.array([1, 0]) * np.exp(-1.0j * xi * (-t_span / 2.0 - dt / 2.0))
        xd[0] = np.array([0, 0]) * -1.0j * (-t_span / 2.0 - dt / 2.0) * np.exp(-1.0j * xi * (-t_span / 2.0 - dt / 2.0))
        # y[-1] = np.array([0, 1]) * np.exp(-1.0j * xi * (t_span / 2.0 + dt / 2.0))
        y[-1] = np.array([0, 1]) * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))

    for k in range(n):

        if type == 'orig':
            a_matrix = np.array([[1.0, dt * q[k] * np.exp(2.0j * xi * t[k])],
                                 [-dt * np.conj(q[k]) * np.exp(-2.0j * xi * t[k]), 1.0]])
            ad_matrix = np.array([[0.0, 2.0j * t[k] * dt * q[k] * np.exp(2.0j * xi * t[k])],
                                  [2.0j * t[k] * dt * np.conj(q[k]) * np.exp(-2.0j * xi * t[k]), 0.0]])
            b_matrix = np.array([[1.0, -dt * q[-k - 1] * np.exp(2.0j * xi * t[-k - 1])],
                                 [dt * np.conj(q[-k - 1]) * np.exp(-2.0j * xi * t[-k - 1]), 1.0]])
        else:
            a_matrix = get_transfer_matrix(q, dt, k, xi, type, sigma)
            ad_matrix = get_transfer_matrix(q, dt, k, xi, type + 'd', sigma)
            b_matrix = get_transfer_matrix(q, -dt, -k - 1, xi, type, sigma)

        x[k + 1] = np.matmul(a_matrix, x[k])
        xd[k + 1] = np.matmul(a_matrix, xd[k]) + np.matmul(ad_matrix, x[k])

        y[-k - 2] = np.matmul(b_matrix, y[-k - 1])

    # print(x[0])

    with warnings.catch_warnings(record=True) as w:

        k_target = 0
        diff = 100
        for k in range(n + 1):

            if type == 'orig':
                value = np.absolute(np.absolute(x[k][0]) - 0.5)
            else:
                value = np.absolute(np.absolute(x[k][0] * np.exp(1.0j * xi * (dt * k - t_span / 2.0 - dt / 2.0))) - 0.5)

            if value < diff:
                k_target = k
                diff = value

    # print(k_target)
    # print(x[k_target][0])
    # print(x[-1])
    # print(x[k_target][0] / y[k_target][0])

    if type == 'orig':
        a = x[-1][0]
        ad = xd[-1][0]
    else:
        # b = x[k_target][0] / y[k_target][0] * np.exp(-1.0j * 2.0 * xi * (t_span / 2.0 + dt / 2.0))
        # b = x[k_target][0] / y[k_target][0] * np.exp(-1.0j * xi * (t_span + dt))  # the closest phase
        # b = x[k_target][0] / y[k_target][0]

        a = x[-1][0] * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))
        ad = xd[-1][0] * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))

    b0 = x[k_target][0] / y[k_target][0] * (1 - np.power(np.absolute(y[k_target][1]), 2)) + x[k_target][1] * np.conj(
        y[k_target][1])

    # b = (x[k_target][0] - a * np.conj(y[k_target][1])) / y[k_target][0]

    b1 = x[k_target][1] / y[k_target][1] * (1 - np.power(np.absolute(y[k_target][0]), 2)) + x[k_target][0] * np.conj(
        y[k_target][0])

    return a, 0.5 * (b0 + b1)


def do_bi_direct_arbitrary_array(q, t, xi, type='orig'):
    """
    Bi-directional algorithm to calculate residues and norming constants for array of discrete spectrum points

    Args:
        q: signal
        t: time grid
        xi: array of spectral parameters where we calculate b(xi) (r(xi))
        type: type of calculation scheme

            - 'orig' -- Ablowitz-Ladik scheme
            - 'bo' -- Bofetta-Osborne
            - 'tes4' -- TES4
            - 'es4' -- ES4

    Returns:
        b: array of b-coefficients (norming constants)
        r: array of r-coefficient (residues)
        ad: array of derivative of a-coefficients in points xi

    """
    n_xi = len(xi)
    a = np.zeros(n_xi, dtype=np.complex128)
    b = np.zeros(n_xi, dtype=np.complex128)
    # r = np.zeros(n_xi, dtype=np.complex128)
    # ad = np.zeros(n_xi, dtype=np.complex128)
    for k in range(n_xi):
        a[k], b[k] = do_bi_direct_arbitrary(q, t, xi[k], type)

    return a, b


def get_pauli_coefficients(m):
    """
    Calculate matrix decomposition coefficients into Pauli matrices
    ::math:`m = a_0 * \\sigma_0 + a_1 * \\sigma_1 + a_2 * \\sigma_2 + a3 * \\sigma_3`

    Args:
        m: 2x2 matrix

    Returns:
        a: decomposition coefficients

    """
    # m = a0 * s0 + a1 * s1 + a2 * s2 + a3 * s3
    # a0 = 1 / 2 * (m[0][0] + m[1][1])
    # a1 = 1 / 2 * (m[0][1] + m[1][0])
    # a2 = 1j / 2 * (m[0][1] - m[1][0])
    # a3 = 1 / 2 * (m[0][0] - m[1][1])

    a = np.zeros(4, dtype=np.complex128)
    a[0] = 1 / 2 * (m[0][0] + m[1][1])
    a[1] = 1 / 2 * (m[0][1] + m[1][0])
    a[2] = 1j / 2 * (m[0][1] - m[1][0])
    a[3] = 1 / 2 * (m[0][0] - m[1][1])

    return a


def expm_2x2(m):
    """
    Calculate exponential function of 2x2 matrix

    Args:
        m: 2x2 matrix

    Returns:
        exponential function of 2x2 matrix

    """
    # Exponential of matrix

    # s0 = np.array([[1, 0], [0, 1]])
    # s1 = np.array([[0, 1], [1, 0]])
    # s2 = -1j * np.array([[0, 1], [-1, 0]])
    # s3 = np.array([[1, 0], [0, -1]])

    # m = a0 * s0 + a1 * s1 + a2 * s2 + a3 * s3
    # a0 = 1 / 2 * (m[0][0] + m[1][1])
    # a1 = 1 / 2 * (m[0][1] + m[1][0])
    # a2 = 1j / 2 * (m[0][1] - m[1][0])
    # a3 = 1 / 2 * (m[0][0] - m[1][1])

    a0 = 1 / 2 * (m[0][0] + m[1][1])
    a1 = 1 / 2 * (m[0][1] + m[1][0])
    a2 = 1j / 2 * (m[0][1] - m[1][0])
    a3 = 1 / 2 * (m[0][0] - m[1][1])

    w = np.sqrt(-a1 ** 2 - a2 ** 2 - a3 ** 2)
    c = np.cos(w)
    if np.absolute(w) > 1e-9:
        s = np.sin(w) / w
    else:
        s = 1.0 - 1. / 6. * w ** 2 + 1. / 120. * w ** 4

    expm = np.exp(a0) * np.array([[c + s * a3, s * (a1 - 1j * a2)], [s * (a1 + 1j * a2), c - s * a3]])

    return expm


def get_transfer_matrix(q, dt, n, xi, type='bo', sigma=1):
    """
    Return transfer matrix for one time-step of scattering problem for Jost function.

    Args:
        q: initial signal
        dt: time step
        n: index number (for q[n]) where calculate transfer matrix
        xi: spectral parameter

    Optional Args:
        type: type of calculation, default = 'bo' -- Bofetta-Osborne

            - bo -- Bofetta-Osborne
            - bod -- Bofetta-Osborne with da / dxi
            - tes4 -- TES4
            - tes4d -- TES4 with da / dxi
            - es4 -- ES4
            - es4d -- ES4 with da / dxi

        sigma: defines focusing (+1) and defocusing (-1) cases


    Returns:
        t_matrix: transfer matrix for index n

    """
    n_q = len(q)
    # Add first element for 4th order schemes
    if n == n_q - 1:
        q = np.append(q, q[0])
    if n < -1:
        n = n_q + n

    if (type == 'bo') | (type == 'bod'):
        k = sp.sqrt(-sigma * np.power(np.absolute(q[n]), 2) - np.power(xi, 2))
        _k = 1.0 / k
        k_m_dt = k * dt
        sinh_k = np.sinh(k_m_dt)
        cosh_k = np.cosh(k_m_dt)

        if type == 'bo':
            t_matrix = np.array([[cosh_k - 1.0j * xi * _k * sinh_k, q[n] * _k * sinh_k],
                                 [-sigma * np.conj(q[n]) * _k * sinh_k, cosh_k + 1.0j * xi * _k * sinh_k]])

        else:

            _k2 = np.power(k, -2)
            xi2_k2 = np.power(xi, 2) * _k2
            t_matrix = np.zeros((2, 2), dtype=np.complex128)
            t_matrix[0][0] = 1.0j * dt * xi2_k2 * cosh_k - (xi * dt + 1.0j * (1.0 + xi2_k2)) * sinh_k * _k
            t_matrix[0][1] = -q[n] * xi * _k2 * (dt * cosh_k - sinh_k * _k)
            t_matrix[1][0] = sigma * np.conj(q[n]) * xi * _k2 * (dt * cosh_k - sinh_k * _k)
            t_matrix[1][1] = -1.0j * dt * xi2_k2 * cosh_k - (xi * dt - 1.0j * (1.0 + xi2_k2)) * sinh_k * _k

    elif (type == 'tes4') | (type == 'tes4d') | (type == 'es4') | (type == 'es4d'):

        q_matrix = np.array([[-1.0j * xi, q[n]],
                             [-sigma * np.conj(q[n]), 1.0j * xi]])

        # Q_1 = (Q_{n + 1} - Q_{n - 1}) / (2 * dt)
        q_1_matrix = np.array([[0.0, q[n + 1] - q[n - 1]],
                               [-sigma * np.conj(q[n + 1] - q[n - 1]), 0.0]])

        # Q_2 = (Q_{n + 1} - 2 * Q_{n} + Q_{n - 1}) / (dt ** 2)
        q_2_matrix = np.array([[0.0, q[n + 1] - 2.0 * q[n] + q[n - 1]],
                               [-sigma * np.conj(q[n + 1] - 2.0 * q[n] + q[n - 1]), 0.0]])

        if type == 'tes4':
            t_matrix = np.matmul(np.matmul(expm_2x2(dt / 24. * q_1_matrix + dt / 48. * q_2_matrix),
                                           expm_2x2(dt * q_matrix)),
                                 expm_2x2(-dt / 24. * q_1_matrix + dt / 48. * q_2_matrix))
        elif type == 'tes4d':

            a = get_pauli_coefficients(q_matrix)
            w = np.sqrt(-a[1] ** 2 - a[2] ** 2 - a[3] ** 2)

            # c = np.cos(w)
            # s = np.sin(w) / w

            # 3rd Pauli matrix
            sigma_3 = np.array([[1, 0],
                                [0, -1]])

            # Central matrix
            if abs(w) < 1e-5:
                sin_wdt = dt * (1.0 - (w * dt) ** 2 / 6.0 + (w * dt) ** 4 / 120.0)
                second_term = dt ** 3 * ((1.0 / 6.0 - 1.0 / 2.0) + (w * dt) ** 2 * (1.0 / 24.0 - 1.0 / 120.0))
                td_matrix = -dt * xi * sin_wdt * np.array([[1, 0], [0, 1]]) + \
                            xi * second_term * q_matrix - 1.0j * sin_wdt * sigma_3
            else:
                # td_matrix = -dt * xi / w * np.sin(w * dt) * np.array([[1, 0], [0, 1]]) + \
                #            xi / w ** 3 * (dt * w * np.cos(w * dt) - np.sin(w * dt)) * q_matrix - \
                #            1.0j * np.sin(w * dt) / w * sigma_3

                td_matrix = -1.0 * np.sin(w * dt) / w * (dt * xi * np.array([[1, 0], [0, 1]]) + 1.0j * sigma_3) + \
                            xi / w ** 3 * (dt * w * np.cos(w * dt) - np.sin(w * dt)) * q_matrix

            t_matrix = np.matmul(np.matmul(expm_2x2(dt / 24. * q_1_matrix + dt / 48. * q_2_matrix),
                                           td_matrix),
                                 expm_2x2(-dt / 24. * q_1_matrix + dt / 48. * q_2_matrix))

        elif type == 'es4':
            f_1_matrix = dt * q_matrix
            f_3_matrix = dt / 24. * q_2_matrix + dt ** 2 / 24. * (np.matmul(q_1_matrix, q_matrix) +
                                                                  np.matmul(q_matrix, q_1_matrix))
            t_matrix = expm_2x2(f_1_matrix)
        else:
            return 0

    else:
        return 0

    return t_matrix


def get_scattering(q, t, xi, type='bo', sigma=1):
    """
    Calculate scattering coefficients with transfer matrices for signal q and spectral parameter xi

    Args:
        q: signal points in time grid
        t: time grid
        xi: spectral parameter where we calculate coefficients

    Optional Args:
        type: type of calculation, default = 'bo' -- Bofetta-Osborne

            - bo -- Bofetta-Osborne
            - bod -- Bofetta-Osborne with :math:`\\partial a(\\xi) / \\partial \\xi`
            - tes4 -- TES4
            - tes4d -- TES4 with :math:`\\partial a(\\xi) / \\partial \\xi`
            - es4 -- ES4
            - es4d -- ES4 with :math:`\\partial a(\\xi) / \\partial \\xi`

        sigma: defines focusing (+1) and defocusing (-1) cases

    Returns:
        (a, b) or (a, b, ad) for 'bod', 'tes4d' and 'es4d'

        - a -- coefficient :math:`a(\\xi)`
        - b -- coefficient :math:`b(\\xi)`
        - ad -- derivative :math:`\\partial a(\\xi) / \\partial \\xi`, optional

    Examples:
        >>>
        >>> get_scattering(q, t, xi, type='bo', sigma=1)


    """
    # We use grid t_n = -t_span / 2. - dt / 2 + dt * n
    # First point is -t_span / 2. - dt / 2
    # Last point is t_span / 2. + dt / 2
    # Full scattering interval is t_span + dt

    n_t = len(q)
    t_span = t[-1] - t[0]  # should be equal to dt * (n_t - 1)
    dt = t[1] - t[0]
    if abs(t_span - dt * (n_t - 1)) > dt / 2.:
        print('[nft_analyse, get_scattering] Error: check t_span definition')
        return 0, 0

    psi = np.array([[1], [0]]) * np.exp(-1.0j * xi * (-t_span / 2.0 - dt / 2.0))
    psi_d = np.array([[1], [0]]) * -1.0j * (-t_span / 2.0 - dt / 2.0) * np.exp(-1.0j * xi * (-t_span / 2.0 - dt / 2.0))

    for n in range(n_t):
        if type[-1] == 'd':
            t_matrix = get_transfer_matrix(q, dt, n, xi, type[0: len(type) - 1], sigma)
            td_matrix = get_transfer_matrix(q, dt, n, xi, type, sigma)
            psi_d = np.matmul(t_matrix, psi_d) + np.matmul(td_matrix, psi)
            psi = np.matmul(t_matrix, psi)
        else:
            t_matrix = get_transfer_matrix(q, dt, n, xi, type, sigma)
            psi = np.matmul(t_matrix, psi)

    a = psi[0] * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))
    b = psi[1] * np.exp(-1.0j * xi * (t_span / 2.0 + dt / 2.0))

    # a = psi[0] * np.exp(1.0j * xi * (t_span / 2.0 - dt / 2.0))
    # b = psi[1] * np.exp(-1.0j * xi * (t_span / 2.0 - dt / 2.0))

    if type[-1] == 'd':
        ad = psi_d[0] * np.exp(1.0j * xi * (t_span / 2.0 + dt / 2.0))

        return a, b, ad

    return a, b


def get_scattering_array(q, t, xi, type='bo', sigma=1):
    """
        Calculate scattering coefficients with transfer matrices for signal q and array of spectral parameters xi
        Args:
            q: signal points in time grid
            t: time grid
            xi: array of spectral parameters where we calculate coefficients

        Optional Args:
            type: type of calculation, default = 'bo' -- Bofetta-Osborne

                - bo -- Bofetta-Osborne
                - bod -- Bofetta-Osborne with da / dxi
                - tes4 -- TES4
                - tes4d -- TES4 with da / dxi
                - es4 -- ES4
                - es4d -- ES4 with da / dxi

            sigma: defines focusing (+1) and defocusing (-1) cases

        Returns:
            Arrays of scattering coefficients (a, b) or (a, b, ad)

    """
    n_xi = len(xi)
    a = np.zeros(n_xi, dtype=np.complex128)
    b = np.zeros(n_xi, dtype=np.complex128)
    if type[-1] == 'd':
        ad = np.zeros(n_xi, dtype=np.complex128)
        for k in range(n_xi):
            a[k], b[k], ad[k] = get_scattering(q, t, xi[k], type, sigma)

        return a, b, ad
    else:
        for k in range(n_xi):
            a[k], b[k] = get_scattering(q, t, xi[k], type, sigma)
            # print(a[k], b[k], xi[k])

        return a, b


def test_nft(ampl, chirp, t_span, n_t, n_grid, type='bo', fnft_type=11, plot_flag=1, function=np.absolute):
    """
    Test function for NFT.
    It calculates and draws nonlinear spectrum for sech shape and compare it with theoretical values
    sech = a * sech(t) ^ (1+ 1j * c)
    Than it shows order of calculation methods

    Args:
        ampl: amplitude for sech shape
        chirp: chirp parameter
        t_span: length of full region in t domain, t in [-t_span/2; t_span/2]
        n_t: number of discretisation points in t domain
        n_grid: number of different n_xi

    Optional Args:
        type: type of calculation, default = 'bo' -- Bofetta-Osborne

                - bo -- Bofetta-Osborne
                - bod -- Bofetta-Osborne with da / dxi
                - tes4 -- TES4
                - tes4d -- TES4 with da / dxi
                - es4 -- ES4
                - es4d -- ES4 with da / dxi

        fnft_type: type of calculation for FNFT library, default = 11
        plot_flag: 1 - plots function(a - a_theory) and same for b, 2 - plots function(a - a_theory) and same for b
        function: function to plot, default = np.absolute()

    Returns:
        nothing

    """
    dt = t_span / (n_t - 1)
    t = np.array([i * dt - t_span / 2. for i in range(n_t)])

    xi_span = np.pi / dt
    n_xi = 2 ** 7
    d_xi = xi_span / (n_xi - 1)
    xi = np.array([i * d_xi - xi_span / 2. for i in range(n_xi)])

    q, a_xi, b_xi, xi_discr, b_discr, r_discr, ad_discr = test_signals.get_sech(t, xi, a=ampl, c=chirp)

    a = np.zeros((n_grid, n_xi), dtype=np.complex128)
    b = np.zeros((n_grid, n_xi), dtype=np.complex128)

    for k in range(n_grid):
        n_t_current = n_t * 2 ** k
        dt_current = t_span / (n_t_current - 1)
        t_current = np.array([i * dt_current - t_span / 2. for i in range(n_t_current)])
        q_current, a_xi_temp, b_xi_temp, xi_discr_temp, b_discr_temp, r_discr_temp, ad_discr_temp = test_signals.get_sech(
            t_current, xi, a=ampl, c=chirp)

        if type == 'bo' or type == 'al' or type == 'es4' or type == 'tes4':
            a_current, b_current = get_scattering_array(q_current, t_current, xi, type)
            a[k] = a_current
            b[k] = b_current
        elif type == 'fnft':
            res = nsev(q_current, t_current, xi[0], xi[-1], n_xi, dst=3, cst=1, dis=fnft_type)
            a[k] = res['cont_a']
            b[k] = res['cont_b']

    if plot_flag == 1:
        matplotlib.rcParams.update({'font.size': 30})

        color = ['red', 'blue', 'green', 'xkcd:light purple', 'xkcd:cyan']
        fig, axs = plt.subplots(3, 1, figsize=(15, 30))
        for k in range(n_grid):
            axs[0].plot(xi, function(a[k] - a_xi), color[k], linewidth=3)
            axs[1].plot(xi, function(b[k] - b_xi), color[k], linewidth=3)
            if k > 0:
                axs[2].plot(xi, np.log2(np.absolute(a[k - 1] - a_xi) / np.absolute(a[k] - a_xi)), color[k], linewidth=3)

        # axs[0].set_xlim(-5, 5)
        # axs[0].set_ylim(0, 20)
        axs[0].set_yscale('log')
        axs[0].set_xlabel(r'$\xi$')
        axs[0].set_ylabel(r'$|a - a_{exact}|$')
        axs[0].grid(True)

        axs[1].set_yscale('log')
        axs[1].set_xlabel(r'$\xi$')
        axs[1].set_ylabel(r'$|b - b_{exact}|$')
        axs[1].grid(True)

        axs[2].set_xlabel(r'$\xi$')
        axs[2].set_ylabel(r'$order$')
        axs[2].grid(True)


    elif plot_flag == 2:
        matplotlib.rcParams.update({'font.size': 30})

        color = ['red', 'blue', 'green', 'xkcd:light purple', 'xkcd:cyan']
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        axs[0].plot(xi, function(a_xi), 'xkcd:indigo', linewidth=7,
                    label='exact')
        axs[1].plot(xi, function(b_xi), 'xkcd:indigo', linewidth=7,
                    label='exact')
        for k in range(n_grid):
            axs[0].plot(xi, function(a[k]), color[k], linewidth=3)
            axs[1].plot(xi, function(b[k]), color[k], linewidth=3)
        # axs[0].set_xlim(-5, 5)
        # axs[0].set_ylim(0, 20)
        axs[0].set_xlabel(r'$\xi$')
        axs[0].set_ylabel(r'$|a|$')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].set_xlabel(r'$\xi$')
        axs[1].set_ylabel(r'$|b|$')
        axs[1].grid(True)
        axs[1].legend()


# @njit
def get_omega_continuous(r, xi, t, direction='left', use_fft=True):
    """
        Calculate kernel of spectrum coefficient r(xi).
        For xi on real axe it corresponds to continuous spectrum part.
        :math:`\\int r(\\xi) e^{i \\xi t} dt`

        Args:
            r: spectrum coefficient r in spectral points xi
            xi: spectral points
            t: time grid

        Returns:
            kernel function

    """
    d_xi = xi[1] - xi[0]
    d_t = t[1] - t[0]
    n_t = len(t)

    if use_fft:

        if direction == 'left':
            # omega_r = fftshift(fft(np.roll(np.conj(r), 0))) / d_t / n_t / 2 / np.pi
            # omega_r = omega_r[::-1]
            # for i in range(len(omega_r)):
            #     if not i % 2:
            #         omega_r[i] = - omega_r[i]
            #
            # omega_r = np.conj(omega_r)

            omega_r = fftshift(ifft(np.roll(np.conj(r), 1))) / d_t

            for i in range(len(omega_r)):
                if i % 2:
                    omega_r[i] = - omega_r[i]
            omega_r = np.conj(omega_r)

        elif direction == 'right':
            omega_r = fftshift(ifft(np.roll(r, 1))) / d_t
            omega_r = omega_r[::-1]
            for i in range(len(omega_r)):
                if i % 2:
                    omega_r[i] = - omega_r[i]

        else:
            print("make_itib_from_scattering: non-existing parameter!")
    else:
        omega_r = np.zeros(n_t, dtype=np.complex128)
        c = 0.5 / np.pi * 0.5 * d_xi
        print(d_xi)
        for j in range(n_t):
            exp_xi_t = np.exp(-1.0j * t[j] * xi)
            x = r * exp_xi_t

            if j == 0:
                print("t[j] =", t[j])
                print("xi =", xi)
                print("r =", r)

            # omega_r[j] = 0.5 / np.pi * trapz(x, dx=d_xi)  # trapz method to integrate

            omega_r[j] = 2 * c * np.sum(x)  # left Riemann sum
            # omega_r[j] = c * (np.sum(x[0:len(x) - 1]) + np.sum(x[1:len(x)]))  # middle Riemann sum
            print(j, omega_r[j])

    return omega_r


@njit
def get_omega_discrete(r, xi, t):
    """
        Calculate kernel of spectrum coefficients r_n(xi_n) for discrete spectrum.

        Args:
            r: spectrum coefficient r_n in discrete spectrum points xi_n
            xi: discrete spectrum points
            t: time grid

        Returns:
            kernel function

        """
    n_xi = len(xi)
    n_t = len(t)
    omega_d = np.zeros(n_t, dtype=np.complex128)
    for j in range(n_xi):
        # omega_d += r[j] * np.exp(-1.0j * t * xi[j])
        omega_d -= 1.0j * r[j] * np.exp(-1.0j * t * xi[j])

    return omega_d


def get_contour_integral(order, contour, a, ad):
    return -0.5j / np.pi * simps(np.power(contour, order) * ad / a, contour)


def get_poly_coefficients(s_values):
    n = len(s_values)
    p_coef = np.zeros(n, dtype=np.complex128)
    p_coef[0] = -s_values[0]
    for i in range(1, n):
        p_coef[i] = -1.0 / (i + 1) * (s_values[i] + np.dot(s_values[:i], p_coef[:i][::-1]))

    return p_coef


def get_roots_contour(q, t, contour, type='bo', a_coefficients=None, ad_coefficients=None):
    if a_coefficients is None:
        a_coefficients, _, ad_coefficients = get_scattering_array(q, t, contour, type + 'd')

    n_discrete = get_contour_integral(0, contour, a_coefficients, ad_coefficients)
    n_discrete = int(np.ceil(n_discrete.real))
    # Interesting fact: if we add 1 to n_discrete (fake root xi = 0), accuracy increases
    # print(n_discrete)

    s_values = np.zeros(n_discrete, dtype=np.complex128)
    for i in range(n_discrete):
        s_values[i] = get_contour_integral(i + 1, contour, a_coefficients, ad_coefficients)

    p_coef = get_poly_coefficients(s_values)
    p_coef = np.concatenate((np.array([1.0]), p_coef))

    roots = eigvals(companion(p_coef))

    return roots


def make_dbp_nft_two_intervals(q_small, t_small, q_big, t_big, z_back, xi_upsampling_small=1, xi_upsampling_big=1,
                               fnft_type_small=0, fnft_type_big=0, inverse_type='both',
                               print_sys_message=False):
    """
    Test function. Do not use it in this version.

    Args:
        q_small:
        t_small:
        q_big:
        t_big:
        z_back:
        xi_upsampling_small:
        xi_upsampling_big:
        fnft_type_small:
        fnft_type_big:
        inverse_type:
        print_sys_message:

    Returns:

    """
    # be careful, z_back is the distance for backward propagation
    # to make forward use -z_back

    # big interval is used to calculate continuous spectrum

    # define xi grid
    n_t_big = len(t_big)
    dt_big = t_big[1] - t_big[0]
    t_span_big = t_big[-1] - t_big[0]
    # print(t_span, dt * (n_t - 1))

    n_t_small = len(t_small)

    n_xi_big = xi_upsampling_big * n_t_big
    n_xi_small = xi_upsampling_small * n_t_small

    rv, xi_val = nsev_inverse_xi_wrapper(n_t_big, t_big[0], t_big[-1], n_xi_big)
    xi_big = xi_val[0] + np.arange(n_xi_big) * (xi_val[1] - xi_val[0]) / (n_xi_big - 1)

    rv, xi_val = nsev_inverse_xi_wrapper(n_t_small, t_small[0], t_small[-1], n_xi_small)
    xi_small = xi_val[0] + np.arange(n_xi_small) * (xi_val[1] - xi_val[0]) / (n_xi_small - 1)

    # make direct nft to calculate continuous spectrum
    res = nsev(q_big, t_big, xi_big[0], xi_big[-1], n_xi_big, dst=3, cst=1, dis=fnft_type_big)
    a = res['cont_a']
    b = res['cont_b']

    # make direct nft to calulate discrete spectrum
    # res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=2, dis=19)
    res = nsev(q_small, t_small, xi_small[0], xi_small[-1], n_xi_small, dst=2, cst=3, dis=fnft_type_small, K=512)
    rd = res['disc_res']
    bd = res['disc_norm']
    xi_d = res['bound_states']
    ad = bd / rd

    if print_sys_message:
        print('Number of discrete eigenvalues:', len(xi_d))

    # this part calculates dnft with other methods (not used yet)
    # bd_test_bid, rd_test_bid, ad_test_bid = nft.do_bi_direct_array(q, t, xi_d, type='tes4')

    # in dimensionless units we have
    # beta2 = -1.0
    # gamma = 1.0

    b_prop = b * np.exp(-2. * 1.0j * z_back * np.power(xi_big, 2))
    bd_prop = bd * np.exp(-2. * 1.0j * z_back * np.power(xi_d, 2))

    coef_t = 2.0  # tib algorithm requires scale for t

    # tib algorithm is very strange
    # for left nft problem it restore right part of the signal
    # to restore left part of the signal we have to use spectral data for right problem,
    # which can be obtain from a,b coefficients for left problem
    # b_right = np.conj(b_left)
    # bd_right = 1.0 / bd_left
    # here right and left devoted to type of dnft, not for right and left part of the signal

    q_total = []
    # restore right q part with itib
    if inverse_type == 'tib' or inverse_type == 'both':
        omega_r = get_omega_continuous(b_prop / a, xi_big, coef_t * t_big)
        if len(xi_d) > 0:
            omega_d = get_omega_discrete(bd_prop / ad, xi_d, coef_t * t_big)
            # omega_d_bid = get_omega_discrete(bd_bid_prop / ad_test_bid, xi_d, coef_t * t)
            q_right_part = make_itib(omega_d + omega_r, coef_t * t_big)
            # q_right_part_bid = make_itib(omega_d_bid + omega_r, coef_t * t)
        else:
            q_right_part = make_itib(omega_r, coef_t * t_big)

        # restore left q part with itib
        omega_r = get_omega_continuous(np.conj(b_prop) / a, xi_big, coef_t * t_big)
        if len(xi_d) > 0:
            omega_d = get_omega_discrete(1.0 / bd_prop / ad, xi_d, coef_t * t_big)
            # omega_d_bid = get_omega_discrete(1.0 / bd_bid_prop / ad_test_bid, xi_d, coef_t * t)
            q_left_part = make_itib(omega_d + omega_r, coef_t * t_big)
            # q_left_part_bid = make_itib(omega_d_bid + omega_r, coef_t * t)
        else:
            q_left_part = make_itib(omega_r, coef_t * t_big)

        q_total = np.concatenate((q_left_part[:len(t_big) // 2], np.conj(q_right_part[:len(t_big) // 2][::-1])))

    q_fnft = []
    # additionally we restore signal with inverse fnft
    if inverse_type == 'fnft' or inverse_type == 'both':
        res = nsev_inverse(xi_big, t_big, b_prop, xi_d, bd_prop / ad, cst=1, dst=0)
        q_fnft = res['q']

    return {'q_total': q_total,
            'q_fnft': q_fnft}


def make_dbp_fnft(q, t, z_back, xi_upsampling=1, inverse_type='both', fnft_type=11, print_sys_message=False):
    """
    Perform NFT backpropagation with FNFT for forward and FNFT/TIB for inverse transform

    Args:
        q: signal at spatial point z
        t: time grid for signal
        z_back: propagation distance z in dimensionless units
        xi_upsampling: upsampling parameter for continuous nonlinear spectrum. len(xi) = xi_upsampling * len(q)
        inverse_type: type for inverse NFT, default = 'both'
            - 'both' -- both fnft (layer peeling with Darboux) and iTIB method
            - 'fnft' -- fnft method (layer peeling with Darboux)
            - 'tib' -- iTIB method, combination for left and right problems
        fnft_type: type of FNFT calculation, default = 11
        print_sys_message: print additional messages during calculation, default = False

    Returns:
        Dictionary with calculated signals in defined spatial point (-z_back)
            - q_total' -- signal calculated with iTIB
            - 'q_fnft' -- signal calculated with fnft
            - 'error' -- result of FNFT procedure

    """
    # be careful, z_back is the distance for backward propagation
    # to make forward use -z_back

    # define xi grid
    n_t = len(t)
    dt = t[1] - t[0]
    t_span = t[-1] - t[0]
    # print(t_span, dt * (n_t - 1))

    n_xi = xi_upsampling * n_t

    rv, xi_val = nsev_inverse_xi_wrapper(n_t, t[0], t[-1], n_xi)
    xi = xi_val[0] + np.arange(n_xi) * (xi_val[1] - xi_val[0]) / (n_xi - 1)

    # make direct nft
    start_time = datetime.now()
    # res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=2, dis=19)
    res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=2, dis=fnft_type, K=1024)
    if print_sys_message:
        print_calc_time(start_time, 'direct NFT')

    if res['return_value'] != 0:
        print('[make_dbp_nft] Error: fnft failed!')
        q_total = np.zeros((len(q)), dtype=np.complex128)
        q_fnft = np.zeros((len(q)), dtype=np.complex128)

        return {'q_total': q_total,
                'q_fnft': q_fnft,
                'error': res['return_value']}

    rd = res['disc_res']
    bd = res['disc_norm']
    r = res['cont_ref']
    a = res['cont_a']
    b = res['cont_b']
    xi_d = res['bound_states']
    ad = bd / rd

    if print_sys_message:
        print('Number of discrete eigenvalues:', len(xi_d))

    # this part calculates dnft with other methods (not used yet)
    # bd_test_bid, rd_test_bid, ad_test_bid = nft.do_bi_direct_array(q, t, xi_d, type='tes4')

    # in dimensionless units we have
    # beta2 = -1.0
    # gamma = 1.0

    b_prop = b * np.exp(-2. * 1.0j * z_back * np.power(xi, 2))
    bd_prop = bd * np.exp(-2. * 1.0j * z_back * np.power(xi_d, 2))

    coef_t = 2.0  # tib algorithm requires scale for t

    # tib algorithm is very strange
    # for left nft problem it restore right part of the signal
    # to restore left part of the signal we have to use spectral data for right problem,
    # which can be obtain from a,b coefficients for left problem
    # b_right = np.conj(b_left)
    # bd_right = 1.0 / bd_left
    # here right and left devoted to type of dnft, not for right and left part of the signal

    q_total = []
    # restore right q part with itib
    if inverse_type == 'tib' or inverse_type == 'both':

        start_time = datetime.now()
        omega_r = get_omega_continuous(b_prop / a, xi, coef_t * t)
        if print_sys_message:
            print_calc_time(start_time, 'continuous part of z-chirp')

        if len(xi_d) > 0:
            start_time = datetime.now()
            omega_d = get_omega_discrete(bd_prop / ad, xi_d, coef_t * t)
            if print_sys_message:
                print_calc_time(start_time, 'discrete part of z-chirp')

            # omega_d_bid = get_omega_discrete(bd_bid_prop / ad_test_bid, xi_d, coef_t * t)

            start_time = datetime.now()
            q_right_part = make_itib(omega_d + omega_r, coef_t * t)
            if print_sys_message:
                print_calc_time(start_time, 'TIB')
            # q_right_part_bid = make_itib(omega_d_bid + omega_r, coef_t * t)
        else:
            start_time = datetime.now()
            q_right_part = make_itib(omega_r, coef_t * t)
            if print_sys_message:
                print_calc_time(start_time, 'TIB')

        # restore left q part with itib

        start_time = datetime.now()
        omega_r = get_omega_continuous(np.conj(b_prop) / a, xi, coef_t * t)
        if print_sys_message:
            print_calc_time(start_time, 'continuous part of z-chirp')

        if len(xi_d) > 0:
            start_time = datetime.now()
            omega_d = get_omega_discrete(1.0 / bd_prop / ad, xi_d, coef_t * t)
            if print_sys_message:
                print_calc_time(start_time, 'discrete part of z-chirp')
            # omega_d_bid = get_omega_discrete(1.0 / bd_bid_prop / ad_test_bid, xi_d, coef_t * t)
            start_time = datetime.now()
            q_left_part = make_itib(omega_d + omega_r, coef_t * t)
            if print_sys_message:
                print_calc_time(start_time, 'TIB')
            # q_left_part_bid = make_itib(omega_d_bid + omega_r, coef_t * t)
        else:
            start_time = datetime.now()
            q_left_part = make_itib(omega_r, coef_t * t)
            if print_sys_message:
                print_calc_time(start_time, 'TIB')

        q_total = np.concatenate((q_left_part[:len(t) // 2], np.conj(q_right_part[:len(t) // 2][::-1])))

    q_fnft = []
    # additionally we restore signal with inverse fnft
    if inverse_type == 'fnft' or inverse_type == 'both':
        res = nsev_inverse(xi, t, b_prop, xi_d, bd_prop / ad, cst=1, dst=0)
        q_fnft = res['q']

    return {'q_total': q_total,
            'q_fnft': q_fnft,
            'error': res['return_value']}

    # additional comments:
    # while left part of the signal always is restored quite good,
    # right part sometimes have ostillations close to 0.
    # It happened because we reordered array q_right with [::-1] and actually problem is on the end of array.
    # We can try to calculate dnft for reordered q[::-1].
    # other idea is to play with coef_t, it can numerically give some gain, but I did not try it yet


def make_dbp_nft(q, t, z_back, xi_upsampling=1,
                 forward_continuous_type='fnft',
                 forward_discrete_type='fnft',
                 forward_discrete_coef_type='fnftpoly',
                 inverse_type='both',
                 fnft_type=0, nft_type='bo',
                 use_contour=False, n_discrete_skip=10,
                 print_sys_message=False):
    """
    Perform NFT backpropagation.

    Args:
        q: signal at spatial point z
        t: time grid for signal
        z_back: propagation distance z in dimensionless units
        xi_upsampling: upsampling parameter for continuous nonlinear spectrum. len(xi) = xi_upsampling * len(q)
        forward_continuous_type: type of calculation of continuous spectrum for forward NFT

            - 'fnft' -- use FNFT to calculate continuous spectrum (real axe only!)
            - 'fnftpoly' -- use polynomials from FNFT to calculate continuous spectrum (arbitrary contour)
            - 'slow' -- use transfer matrices to calculate continuous spectrum (arbitrary contour)

        forward_discrete_type:

            - 'fnft' -- use FNFT to calculate discrete spectrum points (coefficients calculated automatically)
            - 'pjt' -- use PJT (phase jump tracking)
            - 'roots' -- not implemented (other procedures to find polynomial roots -> discrete spectrum points)

        forward_discrete_coef_type:

            - 'fnftpoly' -- use polynomial from FNFT to calculate spectral coefficients (b-coefficient is not stable for
             eigenvalues with large imaginary part)
            - 'bi-direct' -- use bi-directional algorithm (more stable for b-coefficient)

        inverse_type: type for inverse NFT, default = 'both'

            - 'both' -- both fnft (layer peeling with Darboux) and iTIB method
            - 'fnft' -- fnft method (layer peeling with Darboux)
            - 'tib' -- iTIB method, combination for left and right problems

        fnft_type: type of FNFT calculation, default = 0
        nft_type: default = 'bo', type of NFT transfer matricies for slow methods and bi-directional algorithm
        use_contour: default = False, use arbitrary spectral contour in spectral space
        n_discrete_skip: default = 10. If we use contour, how much discrete points rest
        print_sys_message: print additional messages during calculation, default = False

    Returns:
        Dictionary with calculated signals in defined spatial point (-z_back)
            - 'q_total' -- signal calculated with iTIB (combined left and right)
            - 'q_tib_left' -- signal calculated with iTIB (left)
            - 'q_tib_right' -- signal calculated with iTIB (right)
            - 'q_fnft' -- signal calculated with fnft
            -  xi_d' -- discrete spectrum points
            - 'xi' -- spectral points
            - 'b_prop' -- b-coefficients with corresponding propagated phase
            - 'a' -- a-coefficient
            - 'bd_prop' -- b-coefficient with corresponding propagated phase for discrete spectrum
            - 'ad' -- :math:`\\partial a(\\xi) / \\partial \\xi` in discrete spectrum points :math:`\\xi_n`

    """
    # be careful, z_back is the distance for backward propagation
    # to make forward use -z_back

    # define xi grid
    n_t = len(t)
    dt = t[1] - t[0]
    t_span = t[-1] - t[0]
    # print(t_span, dt * (n_t - 1))

    n_xi = xi_upsampling * n_t

    # rv, xi_val = nsev_inverse_xi_wrapper(n_t, t[0], t[-1], n_xi)
    # print(xi_val)
    # xi = xi_val[0] + np.arange(n_xi) * (xi_val[1] - xi_val[0]) / (n_xi - 1)
    # xi_span = xi_val[1] - xi_val[0]

    xi_span = np.pi / dt
    d_xi = xi_span / n_xi
    xi = np.array([(i + 1) * d_xi - xi_span / 2. for i in range(n_xi)])
    print(xi[0], xi[-1])

    # if we want to use root finding procedure for polynomial, we should calculate coefficients
    res_poly = None
    if forward_continuous_type == 'fnftpoly' or forward_discrete_coef_type == 'fnftpoly':
        res_poly = nsev_poly(q, t, dis=fnft_type)

    # here we calculate discrete spectrum
    start_time = datetime.now()
    res_discr = get_discrete_eigenvalues(q, t, type=forward_discrete_type, xi_upsampling=xi_upsampling,
                                         max_discrete_points=2048, res_poly=res_poly)
    if print_sys_message:
        print_calc_time(start_time, 'discrete spectrum')

    xi_d = res_discr['discrete_spectrum']

    if forward_discrete_type == 'fnft' or forward_discrete_type == 'pjt':  # norming constants and residues already calculated
        bd = res_discr['disc_norm']
        rd = res_discr['disc_res']
        ad = bd / rd

    if (forward_discrete_type == 'fnft' and forward_discrete_coef_type != 'fnft') \
            or (forward_discrete_type != 'fnft' and forward_discrete_type != 'pjt'):
        # we have to calculate coefficients
        # even if we have already calculated with fnft but want use other type of calculation of coefficients
        res_discr_coef = get_discrete_spectrum_coefficients(q, t, xi_d, type=forward_discrete_coef_type,
                                                            type_coef='orig', fnft_type=fnft_type, res_poly=res_poly)
        bd = res_discr_coef['bd']
        rd = res_discr_coef['rd']
        ad = res_discr_coef['ad']

    if use_contour:  # find suitable contour for calculation of continuous spectrum
        xi_d, bd, rd, ad, xi = get_cut_spectrum_and_contour(xi_d, bd, ad, rd, n_discrete_skip, xi_span, n_xi)

    res_cont = get_continuous_spectrum(q, t, xi=xi, type=forward_continuous_type, xi_upsampling=xi_upsampling,
                                       fnft_type=fnft_type, nft_type=nft_type,
                                       res_poly=res_poly, coefficient_type='both')

    a = res_cont['a']
    b = res_cont['b']
    b_right = res_cont['b_right']

    if print_sys_message:
        print('Number of discrete eigenvalues:', len(xi_d))


    # in dimensionless units we have
    # beta2 = -1.0
    # gamma = 1.0

    print("b: ", b[0], b[1], b[-2], b[-1])

    b_prop = b * np.exp(-2. * 1.0j * z_back * np.power(xi, 2))
    b_prop_right = b_right * np.exp(2. * 1.0j * z_back * np.power(xi, 2))
    bd_prop = bd * np.exp(-2. * 1.0j * z_back * np.power(xi_d, 2))
    # bd_prop_left = np.conj(b_prop)

    q_tib_total = []
    q_left_part = []
    q_right_part = []

    # restore right q part with itib
    if inverse_type == 'tib' or inverse_type == 'both':
        start_time = datetime.now()

        split_index = int(n_t / 2) + 1

        q_left_part = make_itib_from_scattering(b_prop_right / a, xi, 1.0 / bd_prop / ad, xi_d, t, split_index,
                                                'left', print_sys_message)
        # combine left and right parts
        q_right_part = make_itib_from_scattering(b_prop / a, xi, bd_prop / ad, xi_d, t[::-1], n_t - split_index,
                                                 'right', print_sys_message)
        # restore left q part with itib
        q_tib_total = np.concatenate((q_left_part[:split_index], np.conj(q_right_part[:n_t - split_index][::-1])))

        if print_sys_message:
            print_calc_time(start_time, 'all TIBs')

    q_fnft = []
    # additionally we restore signal with inverse fnft
    if inverse_type == 'fnft' or inverse_type == 'both':
        start_time = datetime.now()
        res = nsev_inverse(np.roll(xi, 0), t, np.roll(b_prop, 0), xi_d, bd_prop, cst=1, dst=0, dis=fnft_type)
        q_fnft = res['q']
        if print_sys_message:
            print_calc_time(start_time, 'inverse FNFT')

    # if it will be required, one could return any other parameters
    return {'q_total': q_tib_total,
            'q_tib_left': q_left_part,
            'q_tib_right': np.conj(q_right_part[::-1]),
            'q_fnft': q_fnft,
            'xi_d': xi_d,
            'xi': xi,
            'b_prop': b_prop,
            'a': a,
            'bd_prop': bd_prop,
            'ad': ad}


def make_dnft(q, t, xi_upsampling=1, fnft_type=0, print_sys_message=False):
    # define xi grid
    n_t = len(t)
    dt = t[1] - t[0]
    t_span = t[-1] - t[0]
    # print(t_span, dt * (n_t - 1))

    n_xi = xi_upsampling * n_t

    rv, xi_val = nsev_inverse_xi_wrapper(n_t, t[0], t[-1], n_xi)
    xi = xi_val[0] + np.arange(n_xi) * (xi_val[1] - xi_val[0]) / (n_xi - 1)

    # make direct nft
    # res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=2, dis=19)
    res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=2, dis=fnft_type, K=1024)

    if res['return_value'] != 0:
        print('[make_dbp_nft] Error: fnft failed!')
        q_total = np.zeros((len(q)), dtype=np.complex128)
        q_fnft = np.zeros((len(q)), dtype=np.complex128)

        return {'q_total': q_total,
                'q_fnft': q_fnft,
                'error': res['return_value']}

    rd = res['disc_res']
    bd = res['disc_norm']
    r = res['cont_ref']
    a = res['cont_a']
    b = res['cont_b']
    xi_d = res['bound_states']
    ad = bd / rd

    result = {'rd': rd,
              'bd': bd,
              'r': r,
              'a': a,
              'b': b,
              'xi_d': xi_d,
              'ad': ad,
              'xi': xi}

    if print_sys_message:
        print('Number of discrete eigenvalues:', len(xi_d))

    return result


def get_continuous_spectrum(q, t, xi=None, type='fnft', xi_upsampling=1, fnft_type=0, nft_type='bo',
                            res_poly=None, coefficient_type='left'):
    """

    Args:
        q: signal on time grid q(t_n)
        t: time grid
        xi: default = None, array of spectral points xi
        type: calculation type

            - 'fnft' -- calculate continuous spectrum on real axe via FNFT methods, fnft_type defines FNFT method
            - 'fnftpoly' -- calculate continuous spectrum on arbitrary axe using polynomial from FNFT, fnft_type defines FNFT method
            - 'slow' -- slow methods, defined by nft_type

        xi_upsampling: upsampling factor for number of points in spectral space.
        fnft_type: type of FNFT calculation
        nft_type: type of slow calculation
        res_poly: default = None. If result for FNFT for nsev_poly has been calculated before, we can use it
        coefficient_type: default = 'left'. Type of scattering problem. Left problem corresponds to Jost function starts from the left of time (-infty)

    Returns:

    """
    n_t = len(t)
    dt = t[1] - t[0]
    # t_span = t[-1] - t[0]

    if xi is None:
        n_xi = xi_upsampling * n_t

        # rv, xi_val = nsev_inverse_xi_wrapper(n_t, t[0], t[-1], n_xi)
        # xi = xi_val[0] + np.arange(n_xi) * (xi_val[1] - xi_val[0]) / (n_xi - 1)

        xi_span = np.pi / dt
        d_xi = xi_span / n_xi
        xi = np.array([i * d_xi - xi_span / 2. for i in range(n_xi)])

    n_xi = len(xi)

    # define zero arrays
    a = np.zeros(n_xi, dtype=np.complex128)
    b = np.zeros(n_xi, dtype=np.complex128)
    r = np.zeros(n_xi, dtype=np.complex128)
    b_right = np.zeros(n_xi, dtype=np.complex128)
    r_right = np.zeros(n_xi, dtype=np.complex128)

    if type == 'fnft':
        # make direct nft
        # this calculates only continuous spectrum on real xi axis
        # to calculate continuous spectrum on an arbitrary contour use 'fnftpoly'
        res = nsev(q, t, xi[0], xi[-1], n_xi, dst=3, cst=1,
                   dis=fnft_type)  # dst=3 -- skip discrete spectrum calculation

        if res['return_value'] != 0:
            print('[get_continuous_spectrum] Error: fnft failed!')

        else:

            a = res['cont_a']
            b = res['cont_b']
            r = b / a
            b_right = np.conj(b)
            r_right = b_right / a

    elif type == 'fnftpoly':

        if res_poly is None:
            res_poly = nsev_poly(q, t, dis=fnft_type)
        a_coef = res_poly['coef_a']
        b_coef = res_poly['coef_b']
        ampl_scale = res_poly['ampl_scale']
        deg = res_poly['deg']
        deg1step = res_poly['deg1step']
        deg1step_denom = res_poly['deg1step_denom']
        phase_a = res_poly['phase_a']  # has to be 0
        phase_b = res_poly['phase_b']

        # transform for spectral parameter z = e ^ (1j * xi * dt)
        z = np.exp(2j * xi / (deg1step - 2 * deg1step_denom) * dt)

        a = np.polyval(ampl_scale * a_coef, z) * np.exp(1j * xi * phase_a)

        if coefficient_type == 'left' or coefficient_type == 'both':
            b = np.polyval(ampl_scale * b_coef, z) * np.exp(1j * xi * phase_b)
            if fnft_type == 49 or fnft_type == 50:  # for Suzuki schemes we have additional multiplication
                a = a * z ** (-2 * n_t)
                b = b * z ** (-2 * n_t)
            r = b / a

        # TODO: check sign for right part in phase and suzuki scheme
        if coefficient_type == 'right' or coefficient_type == 'both':
            b_right = np.polyval(ampl_scale * np.conj(b_coef[::-1]), z) * np.exp(-1j * xi * phase_b)
            if fnft_type == 49 or fnft_type == 50:  # for Suzuki schemes we have additional multiplication
                a = a * z ** (-2 * n_t)
                b_right = b_right * z ** (2 * n_t)
            r_right = b_right / a

    elif type == 'slow':

        a, b = get_scattering_array(q, t, xi, type=nft_type)
        r = b / a

    else:
        print('[get_continuous_spectrum] Error: wrong type')

    return {'a': a, 'b': b, 'r': r, 'b_right': b_right, 'r_right': r_right}


def get_discrete_eigenvalues(q, t, type='fnft', xi_upsampling=1, max_discrete_points=1024, a_cont=None, res_poly=None):
    """
    Calculates discrete spectrum points.

    Args:
        q: signal
        t: time grid
        type: type of calculation

            - 'fnft' -- fnft method for calculating discrete eigenvalues. Additionally, returns norming constants and residues
            - 'pjt' -- Phase Jump Tracking method

        xi_upsampling: upsampling factor to FNFT method, default = 1
        max_discrete_points: number of maximum discrete points to calculate, default = 1024
        a_cont: pre-computed a-coefficient for PJT method to accelerate calculations, default = None

    Returns:
        Dictionary with discrete spectrum points and additional info

        - 'return_fnft' --  return of FNFT method (errors) (only for type='fnft')
        - 'disc_norm' -- norming constants for discrete spectrum (only for type='fnft')
        - 'disc_res' -- residues for discrete spectrum (only for type='fnft')

    Note:
        For type='fnft' it uses fnft_type=0 -- MODAL (AL) scheme
    """
    # types: fnft, pjt

    result = {}
    discrete_spectrum = []

    n_t = len(t)

    if type == 'fnft':

        n_xi = n_t * xi_upsampling
        rv, xi_val = nsev_inverse_xi_wrapper(n_t, t[0], t[-1], n_xi)
        xi = xi_val[0] + np.arange(n_xi) * (xi_val[1] - xi_val[0]) / (n_xi - 1)

        # make direct nft
        # res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=2, dis=19)
        fnft_type = 0  # MODAL -- AL scheme from FNFT
        res = nsev(q, t, xi[0], xi[-1], n_xi, dst=2, cst=3, dis=fnft_type, niter=50, K=max_discrete_points)
        discrete_spectrum = res['bound_states']
        result['return_fnft'] = res['return_value']
        result['disc_norm'] = res['disc_norm']
        result['disc_res'] = res['disc_res']

    elif type == 'pjt':

        res = pjt(q, t, a_cont)
        discrete_spectrum = res['bound_states']
        result['disc_norm'] = res['disc_norm']
        result['disc_res'] = res['disc_res']

    elif type == 'roots':
        # here you can write additional function which return roots of polynomial
        # res_poly contain polynomial coefficients
        if res_poly is not None:
            a_poly_coef = res_poly['cont_a']
        ...

    else:
        print('[get_discrete_eigenvalues]: wrong type parameter')

    result['discrete_spectrum'] = discrete_spectrum

    return result


def get_discrete_spectrum(q, t, type_eigen='fnft', type_coef='orig'):
    """
    Calculate discrete spectrum for given signal q(t) and spectral coefficients for it

    Args:
        q: signal points on time grid t
        t: time grid points
        type_eigen: type of eigenvalue calculation

            - 'fnft' -- calculate spectrum via FNFT methods
            - 'pjt' -- PJT algorithm

        type_coef: type of scattering algorithm for bi-directional algorithm (if type='bi-direct')

            - 'orig' -- Ablowitz-Ladik scheme
            - 'bo' -- Bofetta-Osborne
            - 'tes4' -- TES4
            - 'es4' -- ES4

    Returns:
        Discrete spectrum points and scaterring coefficients

        {'spectrum': spectrum,
        'bd': bd,
        'rd': rd,
        'ad': ad}

    """
    xi_upsampling = 1
    if type_eigen == 'fnft':
        xi_upsampling = 4

    res_discr = get_discrete_eigenvalues(q, t, type=type_eigen, xi_upsampling=xi_upsampling)
    spectrum = res_discr['discrete_spectrum']

    if type_eigen == 'fnft':
        bd = res_discr['disc_norm']
        rd = res_discr['disc_res']
        ad = bd / rd
    elif type_eigen == 'pjt':
        bd, rd, ad = do_bi_direct_array(q, t, spectrum, type=type_coef)
    else:
        print('[get_discrete_spectrum]: wrong type_eigen parameter')

    result = {'spectrum': spectrum,
              'bd': bd,
              'rd': rd,
              'ad': ad}

    return result


def get_discrete_spectrum_coefficients(q, t, discrete_points, type='bi-direct', type_coef='orig', fnft_type=0, res_poly=None):
    """
    Calculates spectral coefficients (::math:`r(\\xi_n)` and ::math:`b(\\xi_n)`) for given discrete spectrum points.

    Args:
        q: signal
        t: time grid
        discrete_points: discrete spectrum points for given signal
        type: type of calculation

            - 'bi-direct' -- bi-directional algorithm
            - 'fnftpoly' -- use polynomial from FNFT to calculate (unstable for b-coefficient)

        type_coef: type of scattering algorithm for bi-directional algorithm (if type='bi-direct')

            - 'orig' -- Ablowitz-Ladik scheme
            - 'bo' -- Bofetta-Osborne
            - 'tes4' -- TES4
            - 'es4' -- ES4

        fnft_type: type of FNFT calculation (if type='fnftpoly')
        res_poly: default = None. Use precomputed results for type='fnftpoly'.

    Returns:
        Dictionary with bd, rd, ad

        {'bd': bd,
        'rd': rd,
        'ad': ad}

    """

    bd = []
    rd = []
    ad = []

    if type == 'bi-direct':
        bd, rd, ad = do_bi_direct_array(q, t, discrete_points, type=type_coef)

    elif type == 'fnftpoly':

        n_t = len(t)
        dt = t[1] - t[0]

        if res_poly is None:
            res_poly = nsev_poly(q, t, dis=fnft_type)
        a_coef = res_poly['coef_a']
        b_coef = res_poly['coef_b']
        ampl_scale = res_poly['ampl_scale']
        deg = res_poly['deg']
        deg1step = res_poly['deg1step']
        deg1step_denom = res_poly['deg1step_denom']
        phase_a = res_poly['phase_a']
        phase_b = res_poly['phase_b']

        ad_coef = np.polyder(a_coef)

        # transform for spectral parameter z = e ^ (1j * xi * dt)
        z = np.exp(2j * discrete_points / (deg1step - 2 * deg1step_denom) * dt)

        # phase_a = 0 -> drop second part of the derivative
        ad = np.polyval(ampl_scale * ad_coef, z) * z * (2j / (deg1step - 2 * deg1step_denom) * dt)
        bd = np.polyval(ampl_scale * b_coef, z) * np.exp(1j * discrete_points * phase_b)
        if fnft_type == 49 or fnft_type == 50:  # for Suzuki schemes we have additional multiplication
            ad = ad * z ** (-2 * n_t) - 2 * n_t * z ** (-2 * n_t - 1) * np.polyval(ampl_scale * a_coef, z) \
                 * z * (2j / (deg1step - 2 * deg1step_denom) * dt)
            bd = bd * z ** (-2 * n_t)

        rd = bd / ad

    else:
        print('[get_discrete_spectrum_coefficients]: wrong type parameter')

    return {'bd': bd,
            'rd': rd,
            'ad': ad}


def get_spectral_contour(discrete_points, n_points_rest, xi_span, n_xi):

    points_sorted = -1j * np.sort_complex(1j * np.array(discrete_points))
    distances = np.imag(points_sorted[:n_points_rest + 1]) - np.imag(points_sorted[1:n_points_rest + 2])

    # initially we use given number of points, but we will see next and previous distances to optimise
    coef = 5.0  # if neighbour distances more than coef times bigger, chose that points
    n_points_optimal = n_points_rest
    if distances[n_points_rest - 2] / distances[n_points_rest - 1] > coef:
        n_points_optimal = n_points_rest - 1
    if distances[n_points_rest] / distances[n_points_rest - 1] > coef:
        n_points_optimal = n_points_rest + 1

    ampl = 0.5 * (np.imag(points_sorted[n_points_optimal] + points_sorted[n_points_optimal + 1]))

    contour = get_raised_contour(ampl, xi_span, n_xi)
    return contour, points_sorted[n_points_optimal]


def get_cut_spectrum_and_contour(xi_d, bd, ad, rd, n_points_rest, xi_span, n_xi):

    xi, xi_d_new = get_spectral_contour(xi_d, n_points_rest, xi_span, n_xi)
    bd_new = []
    ad_new = []
    rd_new = []
    for xi_cur in xi_d_new:
        i_loc = np.where(xi_d == xi_cur)[0][0]
        bd_new.append(bd[i_loc])
        ad_new.append(ad[i_loc])
        rd_new.append(rd[i_loc])

    return xi_d_new, bd, rd, ad, xi


# root finding


def durand_kerner(p, max_iter=10, tol=1e-10, check_tol=False, initial=None, shake=False):

    n_roots = len(p) - 1
    if initial is None:
        initial = np.power(0.4+0.9j, np.arange(n_roots))

    roots = initial

    for iter in range(max_iter):
        for r_ind in range(n_roots):
            # correction =
            product = p[0]
            for r_sub in range(n_roots):
                if r_sub != r_ind:
                    product *= (roots[r_ind] - roots[r_sub])
            roots[r_ind] -= np.polyval(p, roots[r_ind]) / product

        break_flag = False
        if check_tol:
            break_flag = True
            for r_ind in range(n_roots):
                if not break_flag:
                    break
                if np.absolute(np.polyval(p, roots[r_ind])) > tol:
                    break_flag = False

        if break_flag:
            print('break', iter)
            break

    if shake:
        for r_ind in range(n_roots):
            ampl = np.absolute(roots[r_ind])
            roots[r_ind] += 1e-10 * ampl * (np.random.random() + 1.0j * np.random.random())

        roots = durand_kerner(p, max_iter=max_iter, tol=tol, check_tol=check_tol, initial=roots, shake=False)


    return roots


def aberth_ehrlich(p, max_iter=10, tol=1e-10, check_tol=False, initial=None):

    n_roots = len(p) - 1
    if initial is None:
        initial = np.power(0.4+0.9j, np.arange(n_roots))

    dp = np.polyder(p)

    roots = np.copy(initial)

    for iter in range(max_iter):
        for r_ind in range(n_roots):

            p_over_pd = np.polyval(p, roots[r_ind]) / np.polyval(dp, roots[r_ind])
            sum = 0
            for r_sub in range(n_roots):
                if r_sub != r_ind:
                    sum += 1.0 / (roots[r_ind] - roots[r_sub])

            correction = p_over_pd / (1.0 - p_over_pd * sum)
            roots[r_ind] -= correction

        break_flag = False
        if check_tol:
            break_flag = True
            for r_ind in range(n_roots):
                if not break_flag:
                    break
                if np.absolute(np.polyval(p, roots[r_ind])) > tol:
                    break_flag = False

        if break_flag:
            print('break', iter)
            break


    return roots


# Not used

# def find_one_eigenvalue_pjt_cauchy(q, t, xi_cont, a_xi, xi_first, xi_second):
#     eps_stop = 1e-8
#
#     delta_step = np.absolute(xi_first - xi_second)
#     line = [xi_first, xi_second]
#     a_line = []
#
#     res_temp = nsev(q, t, M=2, Xi1=xi_first, Xi2=xi_second, kappa=1, cst=1, dst=2)
#     a_res_temp = res_temp['cont_a']
#     a_line.append(a_res_temp[0])
#     a_line.append(a_res_temp[1])
#
#     step = complex(0., delta_step)
#
#     end = False
#     while not end:
#
#         right_point = line[-1] + step
#         left_point = line[-2] + step
#
#         a_right_point = get_a_cauchy(right_point, xi_cont, a_xi)
#         a_left_point = get_a_cauchy(left_point, xi_cont, a_xi)
#
#         push = False
#         if is_arg_jump(a_left_point, a_right_point) == 1:
#
#             push = True
#             step = left_point - line[-2]
#
#         elif is_arg_jump(a_left_point, a_line[-2]) == 1:
#
#             push = True
#             step = line[-2] - line[-1]
#
#             right_point = left_point
#             left_point = line[-2]
#             a_right_point = a_left_point
#             a_left_point = a_line[-2]
#
#         elif is_arg_jump(a_right_point, a_line[-1]) == 1:
#
#             push = True
#             step = line[-1] - line[-2]
#
#             left_point = right_point
#             right_point = line[-1]
#             a_left_point = a_right_point
#             a_right_point = a_line[-1]
#
#         else:
#             if np.absolute(step) < eps_stop:
#                 print("[find_one_eigenvalue_pjt]: Dest found")
#                 end = True
#             else:
#                 step /= 2.
#
#             if np.absolute(step) < 1e-10:
#                 print("[find_one_eigenvalue_pjt] Error: Step less than 10^-10")
#                 break
#
#         if push:
#             line.append(left_point)
#             line.append(right_point)
#             a_line.append(a_left_point)
#             a_line.append(a_right_point)
#
#     left_point, right_point = fit_phase_jump(xi_cont, a_xi, line[-2], line[-1], accuracy=eps_stop)
#     line.append(left_point)
#     line.append(right_point)
#
#     eigenvalue = (line[-1] + line[-2]) / 2.
#     return eigenvalue
#
#
# def find_one_eigenvalue_pjt(q, t, func, func_args, xi_first, xi_second):
#     eps_stop = 1e-8
#     dt = t[1] - t[0]
#
#     # xi_first, xi_second = fit_phase_jump(q, dt, func, func_args, xi_first, xi_second, accuracy=1e-2)
#
#     delta_step = np.absolute(xi_first - xi_second)
#     line = [xi_first, xi_second]
#     a_line = []
#
#     res_temp = nsev(q, t, M=2, Xi1=xi_first, Xi2=xi_second, kappa=1, cst=1, dst=2)
#     a_res_temp = res_temp['cont_a']
#     a_line.append(a_res_temp[0])
#     a_line.append(a_res_temp[1])
#
#     step = complex(0., delta_step)
#
#     end = False
#     while not end:
#
#         right_point = line[-1] + step
#         left_point = line[-2] + step
#         # print(right_point, left_point)
#
#         a_right_point = func(q, dt, right_point, func_args[0], func_args[1])
#         a_left_point = func(q, dt, left_point, func_args[0], func_args[1])
#
#         push = False
#         if is_arg_jump(a_left_point, a_right_point) == 1:
#
#             push = True
#             step = left_point - line[-2]
#
#         elif is_arg_jump(a_left_point, a_line[-2]) == 1:
#
#             push = True
#             step = line[-2] - line[-1]
#
#             right_point = left_point
#             left_point = line[-2]
#             a_right_point = a_left_point
#             a_left_point = a_line[-2]
#
#         elif is_arg_jump(a_right_point, a_line[-1]) == 1:
#
#             push = True
#             step = line[-1] - line[-2]
#
#             left_point = right_point
#             right_point = line[-1]
#             a_left_point = a_right_point
#             a_right_point = a_line[-1]
#
#         else:
#             if np.absolute(step) < eps_stop:
#                 print("[find_one_eigenvalue_pjt]: Dest found")
#                 end = True
#             else:
#                 step /= 2.
#
#             if np.absolute(step) < 1e-10:
#                 print("[find_one_eigenvalue_pjt] Error: Step less than 10^-10")
#                 break
#
#         if push:
#             line.append(left_point)
#             line.append(right_point)
#             a_line.append(a_left_point)
#             a_line.append(a_right_point)
#
#     left_point, right_point = fit_phase_jump(q, dt, func, func_args, line[-2], line[-1], accuracy=eps_stop)
#     line.append(left_point)
#     line.append(right_point)
#
#     eigenvalue = (line[-1] + line[-2]) / 2.
#     return eigenvalue
#
#
# def find_spectrum_pjt_cauchy(q, t, xi_cont, a_xi):
#     print("--------------- start PJT ---------------")
#
#     spectrum = []
#     time_found = []
#     total_phase_shift, error_code, jump = get_contour_phase_shift_adaptive(q, t, xi_cont, a_xi)
#     if error_code != 0:
#         print("[find_spectrum_pjt] Error: problem with jump localisation")
#         return -1
#     total_discr = round(total_phase_shift / (2. * np.pi))
#     n_jump = len(jump)
#     print("Number of discrete eigenvalues:", total_discr)
#     print("Number of jumps:", n_jump)
#     if n_jump != total_discr:
#         print("[find_spectrum_pjt] Error: problem with jump cleaning")
#         return -2
#
#     for i in range(n_jump):
#         t_start = time.time()
#         spectrum.append(find_one_eigenvalue_pjt(q, t, xi_cont, a_xi, jump[i]['xi'], jump[i]['xi_next']))
#         t_end = time.time()
#         time_found.append(t_end - t_start)
#
#     print("--------------- end PJT ---------------")
#
#     table = PrettyTable()
#     table.field_names = ["n", "eigenvalue", "time, s"]
#
#     soliton_energy = 0.0
#     for i in range(len(spectrum)):
#         table.add_row([i, spectrum[i], time_found[i]])
#         soliton_energy += 4. * spectrum[i].imag
#
#     table.add_row(["-", "-", "-"])
#     table.add_row(["E", soliton_energy, ""])
#
#     print(table)
#
#     return spectrum
#
#
# def find_spectrum_pjt(q, t, func, func_args, xi_cont, a_xi):
#     print("--------------- start PJT ---------------")
#
#     spectrum = []
#     time_found = []
#     total_phase_shift, error_code, jump = get_contour_phase_shift_adaptive(q, t, xi_cont, a_xi)
#     if error_code != 0:
#         print("[find_spectrum_pjt] Error: problem with jump localisation")
#         return -1
#     total_discr = round(total_phase_shift / (2. * np.pi))
#     n_jump = len(jump)
#     print("Number of discrete eigenvalues:", total_discr)
#     print("Number of jumps:", n_jump)
#     if n_jump != total_discr:
#         print("[find_spectrum_pjt] Error: problem with jump cleaning")
#         return -2
#
#     for i in range(n_jump):
#         t_start = time.time()
#         spectrum.append(find_one_eigenvalue_pjt(q, t, func, func_args, jump[i]['xi'], jump[i]['xi_next']))
#         t_end = time.time()
#         time_found.append(t_end - t_start)
#
#     print("--------------- end PJT ---------------")
#
#     table = PrettyTable()
#     table.field_names = ["n", "eigenvalue", "time, s"]
#
#     soliton_energy = 0.0
#     for i in range(len(spectrum)):
#         table.add_row([i, spectrum[i], time_found[i]])
#         soliton_energy += 4. * spectrum[i].imag
#
#     table.add_row(["-", "-", "-"])
#     table.add_row(["E", soliton_energy, ""])
#
#     print(table)
#
#     return spectrum
#
# def fit_phase_jump_cauchy(xi_cont, a_xi, first, second, accuracy=1e-4):
#     direction = second - first
#     a_first = get_a_cauchy(first, xi_cont, a_xi)
#     a_second = get_a_cauchy(second, xi_cont, a_xi)
#
#     i = 1
#     while abs(second - first) > accuracy:
#         middle = first + direction / np.power(2.0, i)
#         a_middle = get_a_cauchy(middle, xi_cont, a_xi)
#
#         if is_arg_jump(a_first, a_second) == 1:
#             a_second = a_middle
#             second = middle
#         else:
#             a_first = a_middle
#             first = middle
#
#         i += 1
#
#         if i >= 100:
#             print("[fit_phase_jump]: Error! Number of iterations more than 100")
#             break
#
#     return first, second
#
#
# def fit_phase_jump(q, dt, func, func_args, first, second, accuracy=1e-4):
#     direction = second - first
#     a_first = func(q, dt, first, func_args[0], func_args[1])
#     a_second = func(q, dt, second, func_args[0], func_args[1])
#
#     i = 1
#     while abs(second - first) > accuracy:
#         middle = first + direction / np.power(2.0, i)
#         a_middle = func(q, dt, middle, func_args[0], func_args[1])
#
#         if is_arg_jump(a_first, a_second) == 1:
#             a_second = a_middle
#             second = middle
#         else:
#             a_first = a_middle
#             first = middle
#
#         i += 1
#
#         if i >= 100:
#             print("[fit_phase_jump]: Error! Number of iterations more than 100")
#             break
#
#     return first, second