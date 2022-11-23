import numpy as np
import hpcom
from ssfm_gpu.ssfm_gpu import dispersion_compensation_manakov, dispersion_compensation


def get_process_interval(signal, n_symb_proc, n_point_symbol, skip=0):
    """

    Args:
        signal:
        n_symb_proc:
        n_point_symbol:
        skip:

    Returns:

    """

    return signal[skip:skip + n_symb_proc * n_point_symbol]


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
    return cut_signal_window(signal_x, signal_parameters, n_symb_proc, n_symb_skip), cut_signal_window(signal_y, signal_parameters, n_symb_proc, n_symb_skip)


def get_default_process_parameters():

    process = {}
    process['z_prop'] = 0
    process['n_symb_proc'] = 1
    process['n_symb_side'] = 1
    process['n_symb_total'] = process['n_symb_proc'] + 2 * process['n_symb_side']
    process['n_symb_skip'] = 0
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

    return process


def get_process_parameters(z_prop, n_symb_proc, n_symb_side, n_symb_skip,
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

    return process


def get_windowed_signal(signal, signal_parameters, process_parameters, channel=None):

    if process_parameters['window_mode'] == 'cdc':

        if channel is None:
            print('for cdc window mode you have to provide channel parameters')
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

        elif signal_parameters['n_polarisations'] == 1:
            signal_cdc = dispersion_compensation(channel, signal, dt)
            signal_cdc_window = cut_signal_window(signal_cdc,
                                                  signal_parameters,
                                                  process_parameters['n_symb_total'],
                                                  process_parameters['n_symb_skip'])

            channel['z_span'] = -channel['z_span']
            signal_proc = dispersion_compensation_manakov(channel, signal_cdc_window[0], signal_cdc_window[1], dt)

        else:
            print('for cdc window mode you have to provide channel parameters')
            return -3

        return signal_proc

    elif process_parameters['window_mode'] == 'plain':

        # do nothing with the signal on the Rx
        return signal

    else:
        print('window mode ' + process_parameters['window_mode'] + ' is not defined')
        return -1


def process_nft_window(signal, signal_parameters, process_parameters):

    # cut signal to process only part of full signal
    signal_process = get_windowed_signal(signal, signal_parameters, process_parameters)

