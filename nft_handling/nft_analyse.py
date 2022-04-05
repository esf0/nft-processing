import sys
# adding signal_handling to the system path
sys.path.insert(0, '../signal_handling/')

from importlib import reload
import FNFTpy
reload(FNFTpy)
from FNFTpy import nsev
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
from ssfm import fiber_propogate, get_soliton_pulse, get_gauss_pulse, get_energy
import matplotlib.pyplot as plt
import matplotlib

import signal_generation as sg

import warnings
from datetime import datetime



reload(sg)
reload(FNFTpy)

from prettytable import PrettyTable


def print_calc_time(start_time, type_of_calc=''):
    end_time = datetime.now()
    total_time = end_time - start_time
    print('Time to calculate ' + type_of_calc, total_time.total_seconds() * 1000, 'ms')


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
    xi = np.zeros(n_vertical * n_horizontal, dtype=complex)
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


def fit_phase_jump_cauchy(xi_cont, a_xi, first, second, accuracy=1e-4):
    direction = second - first
    a_first = get_a_cauchy(first, xi_cont, a_xi)
    a_second = get_a_cauchy(second, xi_cont, a_xi)

    i = 1
    while abs(second - first) > accuracy:
        middle = first + direction / np.power(2.0, i)
        a_middle = get_a_cauchy(middle, xi_cont, a_xi)

        if is_arg_jump(a_first, a_second) == 1:
            a_second = a_middle
            second = middle
        else:
            a_first = a_middle
            first = middle

        i += 1

        if i >= 100:
            print("[fit_phase_jump]: Error! Number of iterations more than 100")
            break

    return first, second


def fit_phase_jump(q, dt, func, func_args, first, second, accuracy=1e-4):
    direction = second - first
    a_first = func(q, dt, first, func_args[0], func_args[1])
    a_second = func(q, dt, second, func_args[0], func_args[1])

    i = 1
    while abs(second - first) > accuracy:
        middle = first + direction / np.power(2.0, i)
        a_middle = func(q, dt, middle, func_args[0], func_args[1])

        if is_arg_jump(a_first, a_second) == 1:
            a_second = a_middle
            second = middle
        else:
            a_first = a_middle
            first = middle

        i += 1

        if i >= 100:
            print("[fit_phase_jump]: Error! Number of iterations more than 100")
            break

    return first, second


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


def find_one_eigenvalue_pjt_cauchy(q, t, xi_cont, a_xi, xi_first, xi_second):
    eps_stop = 1e-8

    delta_step = np.absolute(xi_first - xi_second)
    line = [xi_first, xi_second]
    a_line = []

    res_temp = nsev(q, t, M=2, Xi1=xi_first, Xi2=xi_second, kappa=1, cst=1, dst=2)
    a_res_temp = res_temp['cont_a']
    a_line.append(a_res_temp[0])
    a_line.append(a_res_temp[1])

    step = complex(0., delta_step)

    end = False
    while not end:

        right_point = line[-1] + step
        left_point = line[-2] + step

        a_right_point = get_a_cauchy(right_point, xi_cont, a_xi)
        a_left_point = get_a_cauchy(left_point, xi_cont, a_xi)

        push = False
        if is_arg_jump(a_left_point, a_right_point) == 1:

            push = True
            step = left_point - line[-2]

        elif is_arg_jump(a_left_point, a_line[-2]) == 1:

            push = True
            step = line[-2] - line[-1]

            right_point = left_point
            left_point = line[-2]
            a_right_point = a_left_point
            a_left_point = a_line[-2]

        elif is_arg_jump(a_right_point, a_line[-1]) == 1:

            push = True
            step = line[-1] - line[-2]

            left_point = right_point
            right_point = line[-1]
            a_left_point = a_right_point
            a_right_point = a_line[-1]

        else:
            if np.absolute(step) < eps_stop:
                print("[find_one_eigenvalue_pjt]: Dest found")
                end = True
            else:
                step /= 2.

            if np.absolute(step) < 1e-10:
                print("[find_one_eigenvalue_pjt] Error: Step less than 10^-10")
                break

        if push:
            line.append(left_point)
            line.append(right_point)
            a_line.append(a_left_point)
            a_line.append(a_right_point)

    left_point, right_point = fit_phase_jump(xi_cont, a_xi, line[-2], line[-1], accuracy=eps_stop)
    line.append(left_point)
    line.append(right_point)

    eigenvalue = (line[-1] + line[-2]) / 2.
    return eigenvalue


def find_one_eigenvalue_pjt(q, t, func, func_args, xi_first, xi_second):
    eps_stop = 1e-8
    dt = t[1] - t[0]

    # xi_first, xi_second = fit_phase_jump(q, dt, func, func_args, xi_first, xi_second, accuracy=1e-2)

    delta_step = np.absolute(xi_first - xi_second)
    line = [xi_first, xi_second]
    a_line = []

    res_temp = nsev(q, t, M=2, Xi1=xi_first, Xi2=xi_second, kappa=1, cst=1, dst=2)
    a_res_temp = res_temp['cont_a']
    a_line.append(a_res_temp[0])
    a_line.append(a_res_temp[1])

    step = complex(0., delta_step)

    end = False
    while not end:

        right_point = line[-1] + step
        left_point = line[-2] + step
        # print(right_point, left_point)

        a_right_point = func(q, dt, right_point, func_args[0], func_args[1])
        a_left_point = func(q, dt, left_point, func_args[0], func_args[1])

        push = False
        if is_arg_jump(a_left_point, a_right_point) == 1:

            push = True
            step = left_point - line[-2]

        elif is_arg_jump(a_left_point, a_line[-2]) == 1:

            push = True
            step = line[-2] - line[-1]

            right_point = left_point
            left_point = line[-2]
            a_right_point = a_left_point
            a_left_point = a_line[-2]

        elif is_arg_jump(a_right_point, a_line[-1]) == 1:

            push = True
            step = line[-1] - line[-2]

            left_point = right_point
            right_point = line[-1]
            a_left_point = a_right_point
            a_right_point = a_line[-1]

        else:
            if np.absolute(step) < eps_stop:
                print("[find_one_eigenvalue_pjt]: Dest found")
                end = True
            else:
                step /= 2.

            if np.absolute(step) < 1e-10:
                print("[find_one_eigenvalue_pjt] Error: Step less than 10^-10")
                break

        if push:
            line.append(left_point)
            line.append(right_point)
            a_line.append(a_left_point)
            a_line.append(a_right_point)

    left_point, right_point = fit_phase_jump(q, dt, func, func_args, line[-2], line[-1], accuracy=eps_stop)
    line.append(left_point)
    line.append(right_point)

    eigenvalue = (line[-1] + line[-2]) / 2.
    return eigenvalue


def find_spectrum_pjt_cauchy(q, t, xi_cont, a_xi):
    print("--------------- start PJT ---------------")

    spectrum = []
    time_found = []
    total_phase_shift, error_code, jump = get_contour_phase_shift_adaptive(q, t, xi_cont, a_xi)
    if error_code != 0:
        print("[find_spectrum_pjt] Error: problem with jump localisation")
        return -1
    total_discr = round(total_phase_shift / (2. * np.pi))
    n_jump = len(jump)
    print("Number of discrete eigenvalues:", total_discr)
    print("Number of jumps:", n_jump)
    if n_jump != total_discr:
        print("[find_spectrum_pjt] Error: problem with jump cleaning")
        return -2

    for i in range(n_jump):
        t_start = time.time()
        spectrum.append(find_one_eigenvalue_pjt(q, t, xi_cont, a_xi, jump[i]['xi'], jump[i]['xi_next']))
        t_end = time.time()
        time_found.append(t_end - t_start)

    print("--------------- end PJT ---------------")

    table = PrettyTable()
    table.field_names = ["n", "eigenvalue", "time, s"]

    soliton_energy = 0.0
    for i in range(len(spectrum)):
        table.add_row([i, spectrum[i], time_found[i]])
        soliton_energy += 4. * spectrum[i].imag

    table.add_row(["-", "-", "-"])
    table.add_row(["E", soliton_energy, ""])

    print(table)

    return spectrum


def find_spectrum_pjt(q, t, func, func_args, xi_cont, a_xi):
    print("--------------- start PJT ---------------")

    spectrum = []
    time_found = []
    total_phase_shift, error_code, jump = get_contour_phase_shift_adaptive(q, t, xi_cont, a_xi)
    if error_code != 0:
        print("[find_spectrum_pjt] Error: problem with jump localisation")
        return -1
    total_discr = round(total_phase_shift / (2. * np.pi))
    n_jump = len(jump)
    print("Number of discrete eigenvalues:", total_discr)
    print("Number of jumps:", n_jump)
    if n_jump != total_discr:
        print("[find_spectrum_pjt] Error: problem with jump cleaning")
        return -2

    for i in range(n_jump):
        t_start = time.time()
        spectrum.append(find_one_eigenvalue_pjt(q, t, func, func_args, jump[i]['xi'], jump[i]['xi_next']))
        t_end = time.time()
        time_found.append(t_end - t_start)

    print("--------------- end PJT ---------------")

    table = PrettyTable()
    table.field_names = ["n", "eigenvalue", "time, s"]

    soliton_energy = 0.0
    for i in range(len(spectrum)):
        table.add_row([i, spectrum[i], time_found[i]])
        soliton_energy += 4. * spectrum[i].imag

    table.add_row(["-", "-", "-"])
    table.add_row(["E", soliton_energy, ""])

    print(table)

    return spectrum


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


def make_itib(omega, t, sigma=1):
    # sigma = 1 focusing case
    # sigma = -1 defocusing case
    n_t = len(t)
    dt = t[1] - t[0]

    q = np.zeros(n_t, dtype=complex)
    q[0] = -2.0 * omega[0]

    # y = np.zeros(n_t, dtype=complex)
    # z = np.zeros(n_t, dtype=complex)

    y_prev = np.array([1.0 / (1.0 + sigma * dt ** 2 * np.absolute(omega[0]) ** 2 / 4.0)])
    z_prev = np.array([-0.5 * y_prev[0] * dt * omega[0]])

    beta = np.zeros(n_t, dtype=complex)

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

    q = np.zeros(n_t, dtype=complex)
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


def do_bi_direct(q, t, xi, type='orig', sigma=1):
    # warnings.filterwarnings("error")

    dt = t[1] - t[0]
    n = len(q)
    t_span = t[-1] - t[0]

    x = np.zeros((n + 1, 2), dtype=complex)
    xd = np.zeros((n + 1, 2), dtype=complex)
    y = np.zeros((n + 1, 2), dtype=complex)

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
    n_xi = len(xi)
    b = np.zeros(n_xi, dtype=complex)
    r = np.zeros(n_xi, dtype=complex)
    ad = np.zeros(n_xi, dtype=complex)
    for k in range(n_xi):
        b[k], r[k], ad[k] = do_bi_direct(q, t, xi[k], type)

    return b, r, ad


def get_pauli_coefficients(m):
    # m = a0 * s0 + a1 * s1 + a2 * s2 + a3 * s3
    # a0 = 1 / 2 * (m[0][0] + m[1][1])
    # a1 = 1 / 2 * (m[0][1] + m[1][0])
    # a2 = 1j / 2 * (m[0][1] - m[1][0])
    # a3 = 1 / 2 * (m[0][0] - m[1][1])

    a = np.zeros(4, dtype=complex)
    a[0] = 1 / 2 * (m[0][0] + m[1][1])
    a[1] = 1 / 2 * (m[0][1] + m[1][0])
    a[2] = 1j / 2 * (m[0][1] - m[1][0])
    a[3] = 1 / 2 * (m[0][0] - m[1][1])

    return a


def expm_2x2(m):
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
            t_matrix = np.zeros((2, 2), dtype=complex)
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
    n_xi = len(xi)
    a = np.zeros(n_xi, dtype=complex)
    b = np.zeros(n_xi, dtype=complex)
    if type[-1] == 'd':
        ad = np.zeros(n_xi, dtype=complex)
        for k in range(n_xi):
            a[k], b[k], ad[k] = get_scattering(q, t, xi[k], type, sigma)

        return a, b, ad
    else:
        for k in range(n_xi):
            a[k], b[k] = get_scattering(q, t, xi[k], type, sigma)
            # print(a[k], b[k], xi[k])

        return a, b


def test_nft(ampl, chirp, t_span, n_t, n_grid, type='bo', fnft_type=11, plot_flag=1, function=np.absolute):
    dt = t_span / (n_t - 1)
    t = np.array([i * dt - t_span / 2. for i in range(n_t)])

    xi_span = np.pi / dt
    n_xi = 2 ** 7
    d_xi = xi_span / (n_xi - 1)
    xi = np.array([i * d_xi - xi_span / 2. for i in range(n_xi)])

    q, a_xi, b_xi, xi_discr, b_discr, r_discr, ad_discr = get_sech(t, xi, a=ampl, c=chirp)

    a = np.zeros((n_grid, n_xi), dtype=complex)
    b = np.zeros((n_grid, n_xi), dtype=complex)

    for k in range(n_grid):
        n_t_current = n_t * 2 ** k
        dt_current = t_span / (n_t_current - 1)
        t_current = np.array([i * dt_current - t_span / 2. for i in range(n_t_current)])
        q_current, a_xi_temp, b_xi_temp, xi_discr_temp, b_discr_temp, r_discr_temp, ad_discr_temp = get_sech(
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


def get_omega_continuous(r, xi, t):
    d_xi = xi[1] - xi[0]
    n_t = len(t)
    omega_r = np.zeros(n_t, dtype=complex)
    for j in range(n_t):
        exp_xi_t = np.exp(-1.0j * t[j] * xi)
        # exp_xi_t = np.exp(1.0j * t[j] * xi)
        x = r * exp_xi_t
        # omega_r[j] = 0.5 / np.pi * trapz(x, dx=d_xi)  # trapz method to integrate

        omega_r[j] = 0.5 / np.pi * 0.5 * (np.sum(x[0:len(x) - 1]) + np.sum(x[1:len(x)])) * d_xi  # middle Riemann sum

    return omega_r


# def get_omega_continuous_other(r, xi, t):
#
#     d_xi = xi[1] - xi[0]
#     n_t = len(t)
#     omega_r = np.zeros(n_t, dtype=complex)
#     for j in range(n_t):
#         exp_xi_t = np.exp(-1.0j * t[j] * xi)
#         omega_r[j] = 0.5 / np.pi * trapz(r * exp_xi_t, dx=d_xi)
#
#     return omega_r


def get_omega_discrete(r, xi, t):
    n_xi = len(xi)
    n_t = len(t)
    omega_d = np.zeros(n_t, dtype=complex)
    for j in range(n_xi):
        # omega_d += r[j] * np.exp(-1.0j * t * xi[j])
        omega_d -= 1.0j * r[j] * np.exp(-1.0j * t * xi[j])

    return omega_d


def get_contour_integral(order, contour, a, ad):
    return -0.5j / np.pi * simps(np.power(contour, order) * ad / a, contour)


def get_poly_coefficients(s_values):
    n = len(s_values)
    p_coef = np.zeros(n, dtype=complex)
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

    s_values = np.zeros(n_discrete, dtype=complex)
    for i in range(n_discrete):
        s_values[i] = get_contour_integral(i + 1, contour, a_coefficients, ad_coefficients)

    p_coef = get_poly_coefficients(s_values)
    p_coef = np.concatenate((np.array([1.0]), p_coef))

    roots = eigvals(companion(p_coef))

    return roots


def make_dbp_nft_two_intervals(q_small, t_small, q_big, t_big, z_back, xi_upsampling_small=1, xi_upsampling_big=1,
                               fnft_type_small=11, fnft_type_big=11, inverse_type='both',
                               print_sys_message=False):
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


def make_dbp_nft(q, t, z_back, xi_upsampling=1, inverse_type='both', fnft_type=11, print_sys_message=False):
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
        q_total = np.zeros((len(q)), dtype=complex)
        q_fnft = np.zeros((len(q)), dtype=complex)

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


def make_dnft(q, t, xi_upsampling=1, fnft_type=11, print_sys_message=False):

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
        q_total = np.zeros((len(q)), dtype=complex)
        q_fnft = np.zeros((len(q)), dtype=complex)

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




# Igor's pjt method


@dataclass
class PhaseBreakpoint:
    first: complex
    second: complex
    orientation: float
    derivative_sign: float
    index: int


@dataclass
class SpectralPoint:
    point: complex
    a: complex
    ad: complex
    phase: float
    ad_calculated: bool = False


@dataclass
class Track:
    left: SpectralPoint
    right: SpectralPoint
    step: complex
    orientation: float


def get_spectral_point(xi, q, t, type='bo'):
    if type[-1] == 'd':
        a, _, ad = get_scattering(q, t, xi, type)
        p = SpectralPoint(xi, a, ad, np.angle(a), True)
    else:
        a, _ = get_scattering(q, t, xi, type)
        p = SpectralPoint(xi, a, 0.0, np.angle(a), False)

    return p


def compute_orientation_breakpoint(left, right):
    result = -1
    temp = right - left
    if np.real(temp) > 0 and abs(np.imag(temp)) < 1e-15:
        result = 0
    elif np.imag(temp) > 0 and abs(np.real(temp)) < 1e-15:
        result = 1
    elif np.real(temp) < 0 and abs(np.imag(temp)) < 1e-15:
        result = 2
    elif np.imag(temp) < 0 and abs(np.real(temp)) < 1e-15:
        result = 3

    return result


def determine_phase_breakpoints(contour, a_values, print_sys_message=False):
    # find where phase of a coefficient have a phase jump

    phase_breakpoints = []

    n_contour = len(contour)

    arg_prev = np.angle(a_values[-1])
    arg_values = [arg_prev]

    for i in range(n_contour):
        arg_current = np.angle(a_values[i])

        if check_jump(arg_prev, arg_current):
            arg_values = np.sort(arg_values)

            break_point = PhaseBreakpoint(contour[(i - 1 + n_contour) % n_contour], contour[i],
                                          0, float(arg_prev > 0), i)

            break_point.orientation = compute_orientation_breakpoint(break_point.first, break_point.second)

            phase_breakpoints = phase_breakpoints + [break_point]
            arg_values = []  # TODO: check it

            if print_sys_message:
                print('jump found!', contour[(i - 1 + n_contour) % n_contour], contour[i],
                      ', orientation:', break_point.orientation)

        arg_values = arg_values + [np.angle(a_values[i])]
        arg_prev = arg_current

    return phase_breakpoints


def extend_phase_breakpoints(phase_breakpoints, contour, a_values, max_arg_value=0.25):
    # don't know yet

    phase_breakpoints_extended = phase_breakpoints
    phase_breakpoints_size = len(phase_breakpoints_extended)

    if phase_breakpoints_size > 1:

        for i in range(phase_breakpoints_size):
            coef = 0.06
            if phase_breakpoints_extended[i].orientation == 0:
                coef = 0.03

            # % TODO: check indecies
            max_distance = coef * \
            min(
                abs(phase_breakpoints_extended[i].first -
                    phase_breakpoints_extended[(i - 1 + phase_breakpoints_size) % phase_breakpoints_size].first),
                abs(phase_breakpoints_extended[i].first -
                    phase_breakpoints_extended[(i + 1) % phase_breakpoints_size].first))

            n_contour = len(contour)

            initial_break_point = phase_breakpoints_extended[i]

            left_distance = 0
            right_distance = 0

            # % TODO: check indecies
            for k in range(n_contour):
                ind = (phase_breakpoints_extended[i].index + k + 1) % n_contour
                if np.angle(a_values[ind]) < -max_arg_value and \
                        compute_orientation_breakpoint(phase_breakpoints_extended[i].second,
                                                       contour[ind]) == phase_breakpoints_extended[i].orientation and \
                        abs(initial_break_point.second - contour[ind]) < max_distance:

                    right_distance = abs(contour[ind] - initial_break_point.second)
                else:
                    break

            for k in range(n_contour):
                ind = (phase_breakpoints_extended[i].index - k - 2) % n_contour
                if np.angle(a_values[ind]) > max_arg_value and \
                        compute_orientation_breakpoint(contour[ind],
                                                       phase_breakpoints_extended[i].first) == \
                        phase_breakpoints_extended[i].orientation and \
                        abs(initial_break_point.first - contour[ind]) < max_distance:

                    left_distance = abs(contour[ind] - initial_break_point.first)
                else:
                    break

            phase_breakpoints_extended[i].first = initial_break_point.first - min(left_distance, right_distance)
            phase_breakpoints_extended[i].second = initial_break_point.second + min(left_distance, right_distance)

    return phase_breakpoints_extended


def get_rotation(orientation):
    temp = np.exp(1.0j * (0.5 * np.pi + 2.0 * np.pi * orientation / 4.0))
    return np.round(temp)


def in_rectangle(point, start_point, end_point):
    if point.real < start_point.real or point.real > end_point.real or \
            point.imag < start_point.imag or point.imag > end_point.imag:
        return False
    else:
        return True


def check_jump(first, second):
    # first and second is phases (angles)
    return first > 0 and ((first * second) < 0 and abs(second - first) > 1.2 * np.pi)


def reduce(q, t, type, left_point, right_point, ad_log_abs, limit_value, part=0.5):
    value = get_value_adaptive(left_point.point, right_point.point, ad_log_abs)
    count = 0

    new_border = right_point.point - part * (right_point.point - left_point.point)

    while value > limit_value and count < 10:
        a_value, _, ad_value = get_scattering(q, t, new_border, type)
        new_border_arg_value = np.angle(a_value)

        if count % 2 == 0:
            if left_point.phase * new_border_arg_value <= 0:
                right_point.point = new_border
                right_point.a = a_value

            new_border = left_point.point + part * (right_point.point - left_point.point)
        else:
            if right_point.phase * new_border_arg_value <= 0:
                left_point.point = new_border
                left_point.a = a_value

            new_border = right_point.point - part * (right_point.point - left_point.point)

        value = get_value_adaptive(left_point.point, right_point.point, ad_log_abs)

        count = count + 1

    return left_point, right_point


def refine(q, t, type, left_point, right_point, step, limit_value, orientation, right_point_prev):
    if right_point.ad_calculated:
        ad_log_abs = np.log(abs(right_point.ad))
    else:
        ad_log_abs = get_ad_log_abs(orientation, left_point, right_point, right_point_prev)

    value = get_value_adaptive(left_point.point, right_point.point, ad_log_abs)

    if value > limit_value:
        [left_point, right_point] = reduce(q, t, type, left_point, right_point, ad_log_abs, limit_value, 0.12)
        step = abs(right_point.point - left_point.point)

    return left_point, right_point, step


def get_ad_log_abs(orientation, left_point, right_point, right_point_prev):
    if orientation == 0:
        dfdx = (right_point.a - left_point.a) / np.real(right_point.point - left_point.point)
        dfdy = (right_point.a - right_point_prev.a) / np.imag(right_point.point - right_point_prev.point)
    elif orientation == 1:
        dfdx = (right_point_prev.a - right_point.a) / np.real(right_point_prev.point - right_point.point)
        dfdy = (right_point.a - left_point.a) / np.imag(right_point.point - left_point.point)
    elif orientation == 2:
        dfdx = (left_point.a - right_point.a) / np.real(left_point.point - right_point.point)
        dfdy = (right_point_prev.a - right_point.a) / np.imag(right_point_prev.point - right_point.point)
    else:
        dfdx = (right_point.a - right_point_prev.a) / np.real(right_point.point - right_point_prev.point)
        dfdy = (left_point.a - right_point.a) / np.imag(left_point.point - right_point.point)

    dfdz = 0.5 * (dfdx - 1j * dfdy)
    ad_log_abs = np.log(abs(dfdz))
    return ad_log_abs


def get_value_adaptive(left_point, right_point, ad_log_abs):
    return min(ad_log_abs, -1.0) * abs(right_point - left_point)


def track_to_line(track):
    n_track = len(track)
    track_line = np.zeros(2 * n_track, dtype=complex)
    for n in range(n_track):
        track_line[2 * n] = track[n].left.point
        track_line[2 * n + 1] = track[n].right.point

    return track_line


def pjt_igor(q, t, start_point=None, end_point=None, contour=None, a_values=None, type='bo', print_sys_message=False):
    if not (not (start_point is None or end_point is None) or not (contour is None or a_values is None)):
        print("error: give some arguments please")
        return [], []

    t_span = t[-1] - t[0]
    dt = t[1] - t[0]

    if not (contour is None or a_values is None):
        if print_sys_message:
            print('contour and a_values are given.')
        perimeter = np.sum(np.absolute(contour[1:] - contour[:len(contour) - 1])) + np.absolute(
            contour[-1] - contour[0])

    else:
        if print_sys_message:
            print('contour and a_values will be calculated.')

        perimeter = 2.0 * (end_point.real - start_point.real + end_point.imag - start_point.imag)
        d_xi_border = perimeter * 0.5 / 200  # hyperparameter

        # find breakpoint at the boundary

        d_xi = np.pi * 0.25 / t_span

        n_real = round((end_point.real - start_point.real) / d_xi)
        n_top = round((end_point.real - start_point.real) / d_xi_border)
        n_vertical = round((end_point.imag - start_point.imag) / d_xi_border)
        contour = get_rect(start_point, end_point, n_horizontal=n_real, n_top=n_top, n_vertical=n_vertical)

        # compute a coefficient on border
        t_start = time.time()
        a_values, _ = get_scattering_array(q, t, contour, type)
        t_end = time.time()

        n_total = len(contour)  # total number of points in contour = n_real + n_top + 2 * (n_vertical - 1)

        if print_sys_message:
            print('Time to compute a_coef for contour:', t_end - t_start, 's. n_points =', n_total)

    limit_value = perimeter / 80.0

    # n_total = len(contour)  # total number of points in contour = n_real + n_top + 2 * (n_vertical - 1)

    phase_breakpoints = determine_phase_breakpoints(contour, a_values, print_sys_message)
    phase_breakpoints = extend_phase_breakpoints(phase_breakpoints, contour, a_values)
    n_phase_breakpoints = len(phase_breakpoints)
    discrete_spectrum = np.zeros(n_phase_breakpoints, dtype=complex)

    if print_sys_message:
        print('PJT found', n_phase_breakpoints, 'phase breakpoints')

    track_lines = []

    if n_phase_breakpoints != 0:
        # main part to find track

        for i in range(n_phase_breakpoints):

            current = phase_breakpoints[i]
            if (current.orientation == 0 and (current.second.real > end_point.real or
                                              current.first.real > start_point.real) or
                    current.derivative_sign <= 0):
                continue

            left_point_new = get_spectral_point(current.first, q, t, type)
            right_point_new = get_spectral_point(current.second, q, t, type)

            step = np.absolute(left_point_new.point - right_point_new.point)
            initial_rotation = np.exp(1.0j * 2.0 * np.pi * current.orientation / 4.0)

            # orientation of track:
            # 0 - right, 1 - top, 2 - left, 3 - bottom

            orientation = current.orientation

            left_xi_new = 0.5 * (left_point_new.point + right_point_new.point - step * initial_rotation)
            right_xi_new = 0.5 * (left_point_new.point + right_point_new.point + step * initial_rotation)

            left_point_new = get_spectral_point(left_xi_new, q, t, type)
            right_point_new = get_spectral_point(right_xi_new, q, t, type)

            left_point = left_point_new
            right_point = right_point_new

            # probably unused
            path_length = 0
            step_number = 0

            track = []

            stop_condition = True

            while stop_condition:

                if len(track) > 30000:
                    stop_condition = False
                    print("[nft_analyse, pjt_igor] Error: track is too long!")

                track_point = Track(left_point, right_point, step, orientation)
                track = track + [track_point]

                # Case of the intersection
                track_size = len(track)
                if track_size > 6:
                    orientation_start = track[-6].orientation
                    orientation_finish = track[-1].orientation

                    if orientation_start == (orientation_finish - 1 + 4) % 4:

                        if (
                                orientation_start == track[-5].orientation and
                                orientation_start == track[-4].orientation and
                                orientation_start == track[-3].orientation and
                                orientation_finish == track[-2].orientation or
                                orientation_start == track[-5].orientation and
                                orientation_start == track[-4].orientation and
                                orientation_finish == track[-3].orientation and
                                orientation_finish == track[-2].orientation or
                                orientation_start == track[-7].orientation and
                                orientation_finish == track[-5].orientation and
                                orientation_start == track[-4].orientation and
                                orientation_finish == track[-3].orientation and
                                orientation_finish == track[-2].orientation
                        ):
                            counter = 1

                            for k in range(1, 4):
                                if orientation_finish == track[-k].orientation:
                                    counter += 1

                            orientation = (orientation_finish - 2) % 4
                            rotation = get_rotation(orientation)

                            left_point_suggested = get_spectral_point(track[-1].right.point + step * counter * rotation,
                                                                      q, t, type)
                            right_point_suggested = get_spectral_point(track[-1].left.point + step * counter * rotation,
                                                                       q, t, type)

                            if not np.isfinite(left_point_suggested.a) or not np.isfinite(right_point_suggested.a):
                                stop_condition = 0
                                continue

                            if left_point_suggested.phase * right_point_suggested.phase < 0:
                                left_point = left_point_suggested
                                right_point = right_point_suggested

                                track_point = Track(left_point, right_point, -step, orientation)
                                track = track + [track_point]

                is_refine = True
                calculate_derivative = False  # delete it, we determine it in type

                left_point_prev = left_point
                right_point_prev = right_point

                rotation = get_rotation(orientation)

                right_point_xi_new = right_point.point + step * rotation
                left_point_xi_new = left_point.point + step * rotation

                if not in_rectangle(right_point_xi_new, start_point, end_point) and \
                        not in_rectangle(left_point_xi_new, start_point, end_point):
                    stop_condition = False

                right_point_new = get_spectral_point(right_point_xi_new, q, t, type)

                if check_jump(right_point_new.phase, right_point.phase):
                    right_point_prev = left_point
                    left_point = right_point_new

                    orientation = (orientation - 1 + 4) % 4

                    path_length += step / np.sqrt(2)
                    step_number += 1

                else:
                    left_point_new = get_spectral_point(left_point_xi_new, q, t, type)

                    if check_jump(left_point_new.phase, right_point_new.phase) and stop_condition:
                        left_point = left_point_new
                        right_point = right_point_new

                        path_length = path_length + step
                        step_number = step_number + 1

                    elif check_jump(left_point.phase, left_point_new.phase):
                        left_point_prev = right_point
                        right_point_prev = right_point_new
                        right_point = left_point_new

                        orientation = (orientation + 1) % 4

                        path_length = path_length + step / np.sqrt(2)
                        step_number += 1

                    else:
                        stop_condition = False

                        track_point = Track(left_point_new, right_point_new, step, orientation)
                        track = track + [track_point]

                        probe_point = SpectralPoint(0.0j, 0.0j, 0.0j, 0.0)

                        if len(track) > 2:
                            probe_point.point = 0.25 * (
                                    left_point.point + right_point.point + left_point_new.point + right_point_new.point)
                        else:
                            probe_point.point = 0.5 * (
                                    left_point.point + right_point.point + 1j * end_point.imag * 1e-4)

                        # TODO: check this part
                        # track.emplace_back(probePoint, probePoint, 0, 0, 0, orientation);
                        track_point = Track(probe_point, probe_point, 0, orientation)

                        track = track + [track_point]
                        if print_sys_message:
                            print("Length = ", len(track))

                        # h.discrete_spectrum(end + 1) = probe_point;
                        discrete_spectrum[i] = probe_point.point

                        # break_point_to_discrete_eigenvalue(i) = length(discrete_spectrum) - 1;

                    if stop_condition:
                        if is_refine:
                            left_point, right_point, step = refine(q, t, type, left_point, right_point, step,
                                                             limit_value, orientation, right_point_prev)

            track_lines = track_lines + [track_to_line(track)]
        # tracks{i} = track_to_line(track);
        # plot(real(tracks{i}), imag(tracks{i}))

    # refine_discrete_spectrum(h);

    # disp("Discrete spectrum: " + string(discrete_spectrum));

    print(discrete_spectrum)
    return discrete_spectrum, track_lines
