import numpy as np
import tensorflow as tf
import pandas as pd

from datetime import datetime

from hpcom.signal import create_wdm_parameters, generate_wdm, receiver_wdm, receiver, nonlinear_shift, get_points_wdm, filter_shaper
from hpcom.channel import create_channel_parameters
from hpcom.metrics import get_evm

from ssfm_gpu.propagation import propagate_schrodinger, dispersion_compensation, dispersion_compensation_manakov
from ssfm_gpu.conversion import convert_forward, convert_inverse

# from hpcom.signal import receiver_wdm, receiver, get_points_wdm
import nft_handling.nft_analyse as nft

import FNFTpy as fpy
import PJTpy as pjt


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def add_zeros_to_signal(signal, t, n_add=-1):
    np_signal = len(signal)

    dt = t[1] - t[0]
    if n_add < 0:
        if np_signal % 2 == 1:
            print('[add_zeros_to_signal] Warning: number of points odd! It has to be even!')
        n_add = (next_power_of_2(len(signal)) - len(signal)) // 2

    if n_add == 0:
        return signal, t

    value = 0.0
    signal_new = value * np.ones(np_signal + 2 * n_add, dtype=complex)
    signal_new[n_add: n_add + np_signal] = signal
    # signal_new = sg.add_lateral_function(signal, f_slope, dt, n=n_add)

    t_add = np.arange(n_add) * dt
    t_new = np.concatenate((t_add - t_add[-1] + t[0] - dt, t, t_add + t[-1] + dt), axis=None)

    return signal_new, t_new


def get_interval(signal, n_symb, n_points_symbol, n_symb_skip):
    """

    Args:
        signal:
        n_symb_proc:
        n_points_symbol:
        skip:

    Returns:

    """

    return signal[n_symb_skip * n_points_symbol:(n_symb_skip + n_symb) * n_points_symbol]


def get_sub_signal(signal, signal_parameters, process_parameters):
    signal_cut = get_interval(signal, process_parameters['n_symb_total'], signal_parameters['upsampling'],
                              process_parameters['n_symb_skip'])
    np_signal = len(signal)

    dt = 1. / signal_parameters['sample_freq']
    t_vector = (np.arange(np_signal) - np_signal / 2) * dt
    t_cut = get_interval(t_vector, process_parameters['n_symb_total'], signal_parameters['upsampling'],
                         process_parameters['n_symb_skip'])

    return signal_cut, t_cut


def cut_signal_window(signal, signal_parameters, n_symb_proc, n_symb_skip):
    """

    Args:
        signal: signal to process
        signal_parameters: parameters of the signal
        n_symb_proc: number of symbols in window
        n_symb_skip: number of symbols to skip

    Returns: signal in window

    """
    signal_window = np.zeros(len(signal), dtype=complex)
    n_point_symbol = signal_parameters['upsampling']
    end_point = n_symb_proc * n_point_symbol
    skip = n_symb_skip * n_point_symbol
    signal_window[skip: skip + end_point] = signal[skip: skip + end_point]

    return signal_window


def cut_signal_window_two_polarisation(signal_x, signal_y, signal_parameters, n_symb_proc, n_symb_skip):
    """

    Args:
        signal_x: x-polarisation of the signal
        signal_y: y-polarisation of the signal
        signal_parameters: parameters of the signal
        n_symb_proc: number of symbols in window
        n_symb_skip: number of symbols to skip

    Returns: x- and y-polarisation of signal in window

    """
    return (cut_signal_window(signal_x, signal_parameters, n_symb_proc, n_symb_skip),
            cut_signal_window(signal_y, signal_parameters, n_symb_proc, n_symb_skip))


def get_default_process_parameters():
    process = {}
    process['z_prop'] = 0
    process['n_symb_proc'] = 1
    process['n_symb_side'] = 1
    process['n_symb_total'] = process['n_symb_proc'] + 2 * process['n_symb_side']
    process['n_symb_skip'] = 0
    process['n_symb_add'] = 0
    process['window_mode'] = 'cdc'
    process['xi_upsampling'] = 1
    process['forward_continuous_type'] = 'fnft'
    process['forward_discrete_type'] = 'fnft'
    process['forward_discrete_coef_type'] = 'fnftpoly'
    process['inverse_type'] = 'both'
    process['fnft_type'] = 0
    process['nft_type'] = 'bo'
    process['use_contour'] = False
    process['n_discrete_skip'] = 10  # number of discrete points in spectrum beyond the contour if use_contour=True
    process['print_sys_message'] = False
    process['n_steps'] = 1  # number of processing steps

    return process


def get_process_parameters(z_prop, n_symb_proc, n_symb_side, n_symb_skip,
                           n_symb_add=0,
                           n_steps=1,
                           window_mode='cdc',
                           xi_upsampling=1,
                           forward_continuous_type='fnft',
                           forward_discrete_type='fnft',
                           forward_discrete_coef_type='fnftpoly',
                           inverse_type='both',
                           fnft_type=0, nft_type='bo',
                           use_contour=False, n_discrete_skip=10,
                           print_sys_message=False):
    process = {}
    process['z_prop'] = z_prop
    process['n_symb_proc'] = n_symb_proc
    process['n_symb_side'] = n_symb_side
    process['n_symb_total'] = process['n_symb_proc'] + 2 * process['n_symb_side']
    process['n_symb_add'] = n_symb_add
    process['n_symb_skip'] = n_symb_skip
    process['window_mode'] = window_mode
    process['xi_upsampling'] = xi_upsampling
    process['forward_continuous_type'] = forward_continuous_type
    process['forward_discrete_type'] = forward_discrete_type
    process['forward_discrete_coef_type'] = forward_discrete_coef_type
    process['inverse_type'] = inverse_type
    process['fnft_type'] = fnft_type
    process['nft_type'] = nft_type
    process['use_contour'] = use_contour
    process['n_discrete_skip'] = n_discrete_skip  # number of discrete points in spectrum beyond the contour if use_contour=True
    process['print_sys_message'] = print_sys_message
    process['n_steps'] = n_steps  # number of processing steps

    return process


def get_windowed_signal(signal, signal_parameters, process_parameters, channel=None):

    signal_proc = signal
    if process_parameters['window_mode'] == 'cdc':

        if channel is None:
            print('[get_windowed_signal] Error: for cdc window mode you have to provide channel parameters')
            return -2

        dt = 1. / signal_parameters['sample_freq']

        if signal_parameters['n_polarisations'] == 2:
            signal_cdc = dispersion_compensation_manakov(channel, signal[0], signal[1], dt)
            signal_cdc_window = cut_signal_window_two_polarisation(signal_cdc[0], signal_cdc[1],
                                                                   signal_parameters,
                                                                   process_parameters['n_symb_total'],
                                                                   process_parameters['n_symb_skip'])

            channel['z_span'] = -channel['z_span']
            signal_proc = dispersion_compensation_manakov(channel, signal_cdc_window[0], signal_cdc_window[1], dt)
            channel['z_span'] = -channel['z_span']  # return to original state

        elif signal_parameters['n_polarisations'] == 1:
            signal_cdc = dispersion_compensation(channel, signal, dt)
            signal_cdc_window = cut_signal_window(signal_cdc,
                                                  signal_parameters,
                                                  process_parameters['n_symb_total'],
                                                  process_parameters['n_symb_skip'])

            channel['z_span'] = -channel['z_span']
            signal_proc = dispersion_compensation(channel, signal_cdc_window, dt)
            channel['z_span'] = -channel['z_span']  # return to original state

        else:
            print('[get_windowed_signal] Error: unsupported number of polarisations')
            return -3

        if signal_parameters['n_polarisations'] == 2:
            signal_proc = (signal_proc[0].numpy(), signal_proc[1].numpy())
        else:
            signal_proc = signal_proc.numpy()

    elif process_parameters['window_mode'] == 'plain' or process_parameters['window_mode'] == 'test':

        # do nothing with the signal on the Rx
        if signal_parameters['n_polarisations'] == 2:
            signal_proc = (signal_proc[0].numpy(), signal_proc[1].numpy())
        else:
            signal_proc = signal_proc.numpy()

    else:
        print('window mode ' + process_parameters['window_mode'] + ' is not defined')
        return -1

    # additional number of symbols for sides
    process_parameters['n_symb_skip'] -= process_parameters['n_symb_add']  # shift to the left
    process_parameters['n_symb_total'] += 2 * process_parameters['n_symb_add']  # add two intervals on the sides
    if signal_parameters['n_polarisations'] == 2:
        signal_x_cut, t_x_cut = get_sub_signal(signal_proc[0], signal_parameters, process_parameters)
        signal_y_cut, t_y_cut = get_sub_signal(signal_proc[1], signal_parameters, process_parameters)
        signal_cut = (signal_x_cut, signal_y_cut)
        t_cut = t_x_cut

    elif signal_parameters['n_polarisations'] == 1:

        signal_cut, t_cut = get_sub_signal(signal_proc, signal_parameters, process_parameters)

    else:
        return -1

    process_parameters['n_symb_skip'] += process_parameters['n_symb_add']  # shift to the left
    process_parameters['n_symb_total'] -= 2 * process_parameters['n_symb_add']

    # you have to be careful because you are working with a dictionary, not with the copy of it
    # so changes will affect it in the outer scope

    return signal_cut, t_cut


def process_nft_window(signal, signal_parameters, process_parameters, channel=None):
    # preprocess signal for different window modes
    # e.g. 'plain' window mode do nothing with the signal
    # or 'cdc' compensates dispersion, rests only processing part of the signal and returns dispersion
    # cut signal to process only part of full signal
    # length of the signal corresponds processing parameters
    signal_cut, t_cut = get_windowed_signal(signal, signal_parameters, process_parameters)

    if signal_parameters['n_polarisations'] == 2:
        np_proccesing = len(signal_cut[0])

        # to use NFT we have to round length of the signal to the next power of 2
        # n_add=-1 means round points to the next power of 2,
        # but we can specify it if we need
        signal_for_nft_x, t_for_nft = add_zeros_to_signal(signal_cut[0], t_cut, n_add=-1)
        signal_for_nft_y, t_for_nft = add_zeros_to_signal(signal_cut[1], t_cut, n_add=-1)
        signal_for_nft = (signal_for_nft_x, signal_for_nft_y)
        n_add = (len(signal_for_nft_x) - np_proccesing) // 2  # number of points which was added to the signal
    else:
        np_proccesing = len(signal_cut)
        signal_for_nft, t_for_nft = add_zeros_to_signal(signal_cut, t_cut, n_add=-1)
        n_add = (len(signal_for_nft) - np_proccesing) // 2  # number of points which was added to the signal

    if process_parameters['window_mode'] == 'test':
        dt = 1. / signal_parameters['sample_freq']
        if channel is None:
            print('[process_nft_window] Error: for test case you have to give channel parameters')
            return -2

        if signal_parameters['n_polarisations'] == 2:
            signal_restored = dispersion_compensation_manakov(channel, signal_for_nft[0], signal_for_nft[1], dt)

        elif signal_parameters['n_polarisations'] == 1:
            signal_restored = dispersion_compensation(channel, signal_for_nft, dt)

        else:
            print('[process_nft_window] Error: unsupported number of polarisations')
            return -3

    else:

        if signal_parameters['n_polarisations'] == 1:
            result_dbp_nft = nft.make_dbp_nft(t_for_nft - (t_for_nft[-1] + t_for_nft[0]) / 2,
                                              process_parameters,
                                              signal_for_nft)
        elif signal_parameters['n_polarisations'] == 2:
            result_dbp_nft = nft.make_dbp_nft(t_for_nft - (t_for_nft[-1] + t_for_nft[0]) / 2,
                                              process_parameters,
                                              signal_for_nft[0], signal_for_nft[1])
        else:
            print('[process_nft_window] Error: unknown number of polarisations.')

        if process_parameters['inverse_type'] == 'both' or process_parameters['inverse_type'] == 'tib':
            signal_restored = result_dbp_nft['q_total']
        elif process_parameters['inverse_type'] == 'fnft':
            signal_restored = result_dbp_nft['q_fnft']
        else:
            print("[process_nft_window]: wrong inverse_type")

    print(np.shape(signal_restored), n_add)
    if n_add != 0:
        if signal_parameters['n_polarisations'] == 1:
            signal_restored = signal_restored[n_add:-n_add]  # remove additional zeros on the sides
        elif signal_parameters['n_polarisations'] == 2:
            signal_restored = (signal_restored[0][n_add:-n_add], signal_restored[1][n_add:-n_add])
    print(np.shape(signal_restored))

    result = {}
    # result['n_add'] = n_add
    # result['signal_restored'] = signal_restored
    # result['t_for_nft'] = t_for_nft
    # result['result_dbp_nft'] = result_dbp_nft

    # probably we need only that
    result['signal_restored'] = signal_restored

    return result


def process_wdm_signal(signal, signal_init, channel, wdm_parameters, wdm_info, process_parameters):

    # in case of one polarisation points_y_orig will be empty
    points_x_orig = wdm_info['points_x'][0]  # [0] index for only one WDM channel
    points_y_orig = wdm_info['points_y'][0]  # [0] index for only one WDM channel
    ft_filter_values = wdm_info['ft_filter_values_x']

    points_x_nft = np.array([])
    points_y_nft = np.array([])

    # calculate CDC and get points in that case
    # after that we can compare performance of NFT with CDC
    if wdm_parameters['n_polarisations'] == 1:
        # NLSE case
        signal_cdc = dispersion_compensation(channel, signal, 1. / wdm_parameters['sample_freq'])
        points_x_cdc = get_points_wdm(receiver_wdm(signal_cdc, ft_filter_values, wdm_parameters), wdm_parameters)
        points_x_cdc *= nonlinear_shift(points_x_cdc, points_x_orig)
        points_y_cdc = np.array([])

    elif wdm_parameters['n_polarisations'] == 2:
        # Manakov case
        signal_cdc = dispersion_compensation_manakov(channel, signal[0], signal[1], 1. / wdm_parameters['sample_freq'])
        points_x_cdc = get_points_wdm(receiver_wdm(signal_cdc[0], ft_filter_values, wdm_parameters)[0], wdm_parameters)  # [0] index for only one WDM channel
        points_y_cdc = get_points_wdm(receiver_wdm(signal_cdc[1], ft_filter_values, wdm_parameters)[0], wdm_parameters)
        points_x_cdc *= nonlinear_shift(points_x_cdc, points_x_orig)
        points_y_cdc *= nonlinear_shift(points_y_cdc, points_y_orig)

    else:
        print('[process_wdm_signal] Error: unknown number of polarisations.')
        return -1

    # main loop for window iteration
    # processing step by step full signal
    # number of steps is defined by process_parameters['n_steps']
    for proc_iter in range(process_parameters['n_steps']):

        result_nft = process_nft_window(signal, wdm_parameters, process_parameters)

        # the fastest solution is to add zeros to the end of the resulting signal and find points then cut
        signal_nft = result_nft['signal_restored']

        start_point = process_parameters['n_symb_side']
        end_point = process_parameters['n_symb_side'] + process_parameters['n_symb_proc']
        if process_parameters['window_mode'] == 'cdc':
            start_point -= process_parameters['n_symb_add']  # additional shift for CDC window mode. See explanation

        if wdm_parameters['n_polarisations'] == 1:
            # NLSE case
            signal_nft = np.concatenate([signal_nft, np.zeros(len(signal) - len(signal_nft), dtype=complex)])
            points_x_nft_proc = get_points_wdm(receiver_wdm(signal_nft, ft_filter_values, wdm_parameters)[0], wdm_parameters)

            # put processed points to final result array
            points_x_nft = np.concatenate([points_x_nft, points_x_nft_proc[start_point:end_point]])
            points_y_nft = np.array([])

        elif wdm_parameters['n_polarisations'] == 2:
            # Manakov case
            signal_nft_x = np.concatenate([signal_nft[0], np.zeros(len(signal[0]) - len(signal_nft[0]), dtype=complex)])
            signal_nft_y = np.concatenate([signal_nft[1], np.zeros(len(signal[1]) - len(signal_nft[1]), dtype=complex)])
            points_x_nft_proc = get_points_wdm(receiver_wdm(signal_nft_x, ft_filter_values, wdm_parameters)[0], wdm_parameters)
            points_y_nft_proc = get_points_wdm(receiver_wdm(signal_nft_y, ft_filter_values, wdm_parameters)[0], wdm_parameters)

            # put processed points to final result array
            points_x_nft = np.concatenate([points_x_nft, points_x_nft_proc[start_point:end_point]])
            points_y_nft = np.concatenate([points_y_nft, points_y_nft_proc[start_point:end_point]])

        # shift window to the next processing interval
        process_parameters['n_symb_skip'] += process_parameters['n_symb_proc']

    result = {}
    result['points_x_orig'] = points_x_orig
    result['points_y_orig'] = points_y_orig
    result['points_x_cdc'] = points_x_cdc
    result['points_y_cdc'] = points_y_cdc
    result['points_x_nft'] = points_x_nft
    result['points_y_nft'] = points_y_nft

    return result


def example_nlse_processing(wdm, channel, process_parameters, job_name='test', dir='',
                            display_c_msg=True, omp_num_threads=4, save_flag=False, n_iter_save=1):

    # display_c_msg = False
    # omp_num_threads = 4
    xi_upsampling = 4

    # n_iter_save = 10  # Number of iterations after which to save to a parquet file

    # Initialize DataFrame to hold the variables to save after every N iterations
    intermediate_results = pd.DataFrame()
    k_save = 0

    signal_x, wdm_info = generate_wdm(wdm)
    points_x_orig = wdm_info['points_x'][0]
    ft_filter_values = wdm_info['ft_filter_values_x'][0] / wdm['upsampling']  # [0] index for only one WDM channel

    start_time = datetime.now()
    signal_prop = propagate_schrodinger(channel, signal_x, sample_freq=int(wdm['symb_freq'] * wdm['upsampling']))
    print("propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms")

    signal_cdc = dispersion_compensation(channel, signal_prop, 1. / wdm['sample_freq'])
    points_x_cdc = get_points_wdm(filter_shaper(signal_cdc, ft_filter_values)[::wdm['downsampling_rate']],
                                  wdm)  # downsample
    shift_factor_x = np.dot(np.transpose(np.conjugate(points_x_orig)), points_x_orig) / np.dot(
        np.transpose(np.conjugate(points_x_orig)), points_x_cdc)
    print(shift_factor_x, np.absolute(shift_factor_x))


    # create array to store all recovered points
    points = np.array([])
    points_nft = np.array([])
    evm = np.array([])
    return_values = np.empty((0,3), int)

    for _ in range(process_parameters['n_steps']):
        print("Processing iteration", _, "of", process_parameters['n_steps'])  # print step number

        # Signal windowing
        # ----------------
        # cut the Tx and Rx signals to the size of the window
        signal_windowed_cdc, t_window_cdc = get_windowed_signal(signal_prop, signal_parameters=wdm, process_parameters=process_parameters, channel=channel)
        signal_tx_windowed_cdc, t_tx_window_cdc = get_windowed_signal(signal_x, signal_parameters=wdm, process_parameters=process_parameters, channel=channel)


        # Convert signal to dimensionless form
        # ------------------------------------
        convert_result = convert_inverse(signal_windowed_cdc, t_window_cdc, channel['z_span'],
                                     channel['beta2'],
                                     channel['gamma'],
                                     t0=1. / wdm['symb_freq'], type='nlse')

        q_windowed_cdc = convert_result['q']
        t_dl_window_cdc = convert_result['t']
        dt_dl = t_dl_window_cdc[0] - t_dl_window_cdc[1]
        z_span_dimless = convert_result['z']
        z_dimless = z_span_dimless * channel['n_spans']


        # Arrange parameters for the transformation
        # -----------------------------------------
        t = (np.arange(len(signal_prop)) - len(signal_prop) / 2) / wdm['sample_freq']
        n_t = len(t_dl_window_cdc)
        t_span = t[-1] - t[0]

        # shift time to shift signal in the center of the time interval
        t_for_nft = t_dl_window_cdc - (t_dl_window_cdc[-1] + t_dl_window_cdc[0]) / 2

        # arrange nonlinear frequency grid \xi
        # xi_upsampling = 4
        n_xi = xi_upsampling * n_t
        rv, xi_val = fpy.nsev_inverse_xi_wrapper(n_t, t_for_nft[0], t_for_nft[-1], n_xi)
        xi = xi_val[0] + np.arange(n_xi) * (xi_val[1] - xi_val[0]) / (n_xi - 1)


        # Forward NFT
        # -----------
        fnft_type=11

        start_time = datetime.now()
        result_nlse = fpy.nsev(q_windowed_cdc, t_for_nft, xi[0], xi[-1], n_xi, dst=3, cst=2, dis=fnft_type, K=2048, display_c_msg=display_c_msg)
        print("forward NFT took", (datetime.now() - start_time).total_seconds() * 1000, "ms")
        start_time = datetime.now()
        result_pjt = pjt.pjt(q_windowed_cdc, t_for_nft,
                             contSpec=None, omp_num_threads=omp_num_threads, display_c_msg=display_c_msg)
        # result_pjt = pjt.pjt(q_windowed_cdc, t_for_nft, contSpec=np.concatenate([xi, result_nlse['cont_a']]), omp_num_threads=4, display_c_msg=True)
        print("PJT took", (datetime.now() - start_time).total_seconds() * 1000, "ms")


        # Nonlinear spectrum evolution
        # ----------------------------
        a = result_nlse['cont_a']
        b = result_nlse['cont_b']
        r = result_nlse['cont_ref']

        res = result_pjt

        rd = res['disc_res']
        bd = res['disc_norm']
        xi_d = res['bound_states']
        # ad = bd / rd


        b_prop = b * np.exp(-2. * 1.0j * z_dimless * np.power(xi, 2))
        bd_prop = bd * np.exp(-2. * 1.0j * z_dimless * np.power(xi_d, 2))


        # Inverse NFT
        # -----------
        start_time = datetime.now()
        result_nsev_inverse = fpy.nsev_inverse(xi, t_for_nft, b_prop, xi_d, bd_prop,
                                               cst=1, dst=0, display_c_msg=display_c_msg)
        print("inverse NFT took", (datetime.now() - start_time).total_seconds() * 1000, "ms")
        q_fnft = result_nsev_inverse['q']

        # print("RETURN VALUES FROM FNFT / PJT / IFNFT",
        #       result_nlse['return_value'],
        #       res['return_value'],
        #       result_nsev_inverse['return_value'])

        # Inverse conversion from dimensionless form
        # ------------------------------------------
        convert_forward_result = convert_forward(q_fnft, t_dl_window_cdc, z_span_dimless,
                                             channel['beta2'],
                                             channel['gamma'],
                                             t0=1. / wdm['symb_freq'], type='nlse')

        signal_nft = convert_forward_result['Q']
        t_nft = convert_forward_result['T']


        # Decode constellation points
        # ---------------------------
        # add zeros to the end of the signal to process with the same length as the original signal
        n_signal_total = len(signal_prop)
        signal_nft_complete = np.concatenate([signal_nft, np.zeros(n_signal_total - len(signal_nft))])
        signal_nft_complete = tf.cast(signal_nft_complete, tf.complex128)

        points_x_nft = get_points_wdm(filter_shaper(signal_nft_complete, ft_filter_values)[::wdm['downsampling_rate']], wdm)

        # for NFT processed we have n_symb_add + n_symb_side points from left
        n_points_start_nft = int(process_parameters['n_symb_side'] + process_parameters['n_symb_add'])
        n_points_end_nft = int(n_points_start_nft + process_parameters['n_symb_proc'])  # restore n_symb_proc points
        range_points_nft = range(n_points_start_nft, n_points_end_nft)

        # for original signal we have n_symb_skip + n_symb_side points from left
        n_points_start_orig = int((process_parameters['n_symb_skip'] + process_parameters['n_symb_side']))
        n_points_end_orig = int(n_points_start_orig + process_parameters['n_symb_proc'])  # restore n_symb_proc points
        range_points_orig = range(n_points_start_orig, n_points_end_orig)

        # points_found = scale_p * points_x_window[range_points]
        points_found_nft = points_x_nft[range_points_nft]
        points_was_orig = points_x_orig[range_points_orig]

        # shift and rotate points, if needed
        shift_factor_x_nft = np.dot(np.transpose(np.conjugate(points_was_orig)), points_was_orig) / np.dot(
            np.transpose(np.conjugate(points_was_orig)), points_found_nft)
        points_found_nft *= shift_factor_x_nft
        print("shift factor for x:", shift_factor_x_nft)


        # Now we can store points and continue signal processing
        # ------------------------------------------------------
        points = np.concatenate([points, points_was_orig])
        points_nft = np.concatenate([points_nft, points_found_nft])
        evm = np.append(evm, get_evm(points_was_orig, points_found_nft))
        return_values = np.append(return_values, [[result_nlse['return_value'], res['return_value'], result_nsev_inverse['return_value']]])

        # Now we can increment 'n_symb_skip' by 'n_symb_proc' and continue signal processing
        process_parameters['n_symb_skip'] += process_parameters['n_symb_proc']


        # Create a dictionary with current values of b, r, rd, bd, xi_d, xi, q_windowed_cdc, and t_for_nft
        iteration_data = {
            'b': [b],
            'r': [r],
            'rd': [rd],
            'bd': [bd],
            'xi_d': [xi_d],
            'xi': [xi],
            'q_windowed_cdc': [q_windowed_cdc],
            't_for_nft': [t_for_nft]
        }

        # Append the dictionary as a new row to the DataFrame
        # intermediate_results = intermediate_results.append(iteration_data, ignore_index=True)
        intermediate_results = pd.concat([intermediate_results, pd.DataFrame(iteration_data)], ignore_index=True)

        # If the current iteration number is divisible by N, save the DataFrame to a parquet file
        if save_flag and ((_ % n_iter_save == 0) or (_ == process_parameters['n_steps'] - 1)):
            # intermediate_results.to_parquet(f'results_processing' + job_name + '.parquet')
            intermediate_results.to_pickle(dir + 'results_processing_' + job_name + '_' + str(k_save) + '.pkl')
            intermediate_results = pd.DataFrame()  # clear the DataFrame to save memory
            k_save += 1



    result = {
        'points': points,
        'points_nft': points_nft,
        'evm': evm,
        'return_values': return_values
    }

    # Store the variables in a dictionary
    results_to_save = {
        'signal_x': [signal_x.numpy()],
        'wdm_info': [wdm_info],
        'signal_prop': [signal_prop.numpy()],
        'signal_cdc': [signal_cdc.numpy()],
        'points_x_cdc': [points_x_cdc],
        'points_x_orig': [points_x_orig],
        'process_parameters': [process_parameters],
        'wdm': [wdm],
        'channel': [channel],
        'points': [points],
        'points_nft': [points_nft],
        'evm': [evm],
        'return_values': [return_values]
    }

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(results_to_save)

    # Save the DataFrame to a parquet file
    if save_flag:
        # df.to_parquet('results_meta_' + job_name + '.parquet')
        df.to_pickle(dir + 'results_meta_' + job_name + '.pkl')

    return result