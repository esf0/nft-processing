import numpy as np
import signal_handling.signal_generation as sg
import nft_handling.nft_analyse as nft

from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift


# slope for boundaries interpolation
def slope_old(x):
    if np.absolute(x) > 0:
        return np.exp(-np.power(0.01 * x, 2))
        # return 0.0
    else:
        return 0.0


# def slope(x, param):
#     dy = param[1]
#     dx = param[0]
#     y_scale = param[2]
#     x_shift = param[3]
#
#     ampl = (dy / dx * (x + x_shift) + np.absolute(y_scale))
#     if ampl < 0.0:
#         ampl = 0.0
#     value = ampl * np.exp(1j * np.angle(y_scale))
#
#     return value
#     # if np.absolute(x + 1) > 0:
#     #     return dy / dx * x + 1 - dy
#     # else:
#     #     return 0.0


def slope_zero(x, param):
    return 0.0


def slope_t(t):
    return t


def get_process_interval(signal, n_symb, t_proc, skip=0):
    end_point = int(t_proc * n_symb)
    # print(end_point)
    return signal[skip:skip + end_point]


def get_sub_signal(signal, n_symb, t_symb, num_symb_proc, num_symb_skip, n_shift):
    t_proc = num_symb_proc * t_symb
    signal_cut = get_process_interval(signal, n_symb, t_proc, skip=n_shift + num_symb_skip * n_symb)
    np_signal = len(signal)
    dt = t_symb / n_symb
    t_vector = np.array([(i - np_signal / 2) * dt for i in range(np_signal)])
    t_cut = get_process_interval(t_vector, n_symb, t_proc, skip=n_shift + num_symb_skip * n_symb)

    return signal_cut, t_cut


def cut_signal_window(signal, n_symb, t_symb, num_symb_proc, num_symb_skip, n_shift):
    signal_window = np.zeros(len(signal), dtype=complex)

    t_proc = num_symb_proc * t_symb
    end_point = int(t_proc * n_symb)
    skip = n_shift + num_symb_skip * n_symb
    signal_window[skip: skip + end_point] = signal[skip: skip + end_point]

    return signal_window


def add_lateral_to_signal(signal, f_slope, t, n_add=-1):
    dt = t[1] - t[0]
    if n_add < 0:
        n_add = (sg.next_power_of_2(len(signal)) - len(signal)) // 2

    if n_add == 0:
        return signal, t

    signal_new = sg.add_lateral_function(signal, f_slope, dt, n=n_add)

    t_add = np.array([i * dt for i in range(n_add)])
    # print(n_add)
    t_new = np.concatenate((t_add - t_add[-1] + t[0] - dt, t, t_add + t[-1] + dt), axis=None)

    return signal_new, t_new


# def cut(q, np_symb, ns_proc, ns_d, ns_skip, n_add=0, t_symb=1.):
#     # q -- signal to cut
#     # np_symb -- number of points in one symbol
#     # t_symb -- dimensionless time per one symbol (default = 1)
#     # n_add -- additional number of points from the beginning (default = 0)
#
#     ns_total = 2 * ns_d + ns_proc  # total number of symbols for processing
#
#     # Cut part of the signal to process with FNFT
#     signal_cut, t_cut = get_sub_signal(q, np_symb, t_symb, ns_total, ns_skip, n_add)
#     n_add_cut = (sg.round_power_of_2(1, len(signal_cut)) - len(signal_cut)) // 2
#     signal_proc, t_proc = add_lateral_to_signal(signal_cut, slope, t_cut, n_add_cut)
#
#     return signal_proc, t_proc


def process_fnft_window(q, z_prop, np_symb, t_symb, ns_proc, ns_d, ns_skip, n_shift, window_mode='cdc', beta2=-1.0,
                        xi_upsampling=1, fnft_type=0, print_sys_message=False):
    # q -- singal at Rx
    # z_prop -- distance in dimensionless units

    dt = t_symb / np_symb

    n_t = len(q)
    t_span = (n_t - 1) * dt

    ns_total = 2 * ns_d + ns_proc  # total number of symbols for processing

    if window_mode == 'cdc':

        # make dispersion compensation
        w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
        q_cdc = sg.dispersion_compensation_t(q, w, z_prop, beta2)

        # get window cut for CD compensated signal
        # q_cdc_proc, t_cdc_proc = cut(q_cdc, np_symb, ns_proc, ns_d, ns_skip, t_symb=t_symb)

        # cut CD-compensated signal and inverse CDC (return to propagated signal)
        q_cdc_window = cut_signal_window(q_cdc, np_symb, t_symb, ns_total, ns_skip, n_shift)
        q_proc = sg.dispersion_compensation_t(q_cdc_window, w, -z_prop, beta2)

    elif window_mode == 'plain':

        # do nothing with the signal on the Rx
        q_proc = q

    else:
        print('window mode ' + window_mode + ' is not defined')
        return -1

    q_cut, t_cut = get_sub_signal(q_proc, np_symb, t_symb, ns_total, ns_skip, n_shift)

    # add slope function to side intervals
    # you can define your own slope function
    # default: slope_zero -- fills side intervals with zeros

    n_add_cut = (sg.round_power_of_2(2, len(q_cut)) - len(q_cut)) // 2
    q_for_nft, t_for_nft = add_lateral_to_signal(q_cut, slope_zero, t_cut, n_add_cut)

    result_dbp_nft = nft.make_dbp_fnft(q_for_nft, t_for_nft - (t_for_nft[-1] + t_for_nft[0]) / 2, z_prop,
                                       xi_upsampling=xi_upsampling, inverse_type='tib', fnft_type=fnft_type,
                                       print_sys_message=print_sys_message)

    q_nft = result_dbp_nft['q_total']
    # q_nft = result_dbp_nft['q_fnft']

    return q_nft, t_for_nft, result_dbp_nft, n_add_cut


def process_nft_window(q, z_prop, np_symb, t_symb, ns_proc, ns_d, ns_skip, n_shift, beta2=-1.0,
                       window_mode='cdc',
                       xi_upsampling=1,
                       forward_continuous_type='fnft',
                       forward_discrete_type='fnft',
                       forward_discrete_coef_type='fnftpoly',
                       inverse_type='both',
                       fnft_type=0, nft_type='bo',
                       use_contour=False, n_discrete_skip=10,
                       print_sys_message=False):
    """
    Process given signal with NFT.

    Args:
        q: propagated signal at Rx (receiver)
        z_prop: dimensionless propagation distance
        np_symb: number of points per symbol
        t_symb: dimensionless symbol time interval. Time length of one symbol
        ns_proc: number of symbols to process
        ns_d: number of symbols on the left and right side of processing interval which contain information about chromatic dispersion
        ns_skip: number of symbols from the left to skip
        n_shift: number of additional points on the left and right sides of the signal. See README.md for explanation.
        beta2: :math:`\\beta_2` parameter. +-1
        window_mode: (default = 'cdc') type of window to cut the signal

            - 'cdc' -- CDC-Windowed mode. 1. Compensate chromatic dispersion. 2. Cut signal. 3. De-compensate (return) chromatic dispersion. 4. Process signal
            - 'plain' -- cut signal without any manipulation with it.

        xi_upsampling: (default = 1) upsampling in nonlinear domain. n_xi = xi_upsampling * n_t
        forward_continuous_type: type of calculation of continuous spectrum for forward NFT

            - 'fnft' -- use FNFT to calculate continuous spectrum (real axe only!)
            - 'fnftpoly' -- use polynomials from FNFT to calculate continuous spectrum (arbitrary contour)
            - 'slow' -- use transfer matrices to calculate continuous spectrum (arbitrary contour)

        forward_discrete_type:

            - 'fnft' -- use FNFT to calculate discrete spectrum points (coefficients calculated automatically)
            - 'pjt' -- use PJT (phase jump tracking)
            - 'roots' -- not implemented (other procedures to find polynomial roots -> discrete spectrum points)

        forward_discrete_coef_type:

            - 'fnftpoly' -- use polynomial from FNFT to calculate spectral coefficients (b-coefficient is not stable for eigenvalues with bit imaginary part)
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
        Restored signal and some additional parameters (q_nft, t_for_nft, result_dbp_nft, n_add_cut).
        If inverse_type='both' it uses 'tib'. So better specify directly 'tib' or 'fnft' for inverse NFT.

    """
    # q -- singal at Rx
    # z_prop -- distance in dimensionless units

    dt = t_symb / np_symb

    n_t = len(q)
    t_span = (n_t - 1) * dt

    ns_total = 2 * ns_d + ns_proc  # total number of symbols for processing

    if window_mode == 'cdc':

        # make dispersion compensation
        w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
        q_cdc = sg.dispersion_compensation_t(q, w, z_prop, beta2)

        # get window cut for CD compensated signal
        # q_cdc_proc, t_cdc_proc = cut(q_cdc, np_symb, ns_proc, ns_d, ns_skip, t_symb=t_symb)

        # cut CD-compensated signal and inverse CDC (return to propagated signal)
        q_cdc_window = cut_signal_window(q_cdc, np_symb, t_symb, ns_total, ns_skip, n_shift)
        q_proc = sg.dispersion_compensation_t(q_cdc_window, w, -z_prop, beta2)

    elif window_mode == 'plain':

        # do nothing with the signal on the Rx
        q_proc = q

    else:
        print('window mode ' + window_mode + ' is not defined')
        return -1

    q_cut, t_cut = get_sub_signal(q_proc, np_symb, t_symb, ns_total, ns_skip, n_shift)

    # add slope function to side intervals
    # you can define your own slope function
    # default: slope_zero -- fills side intervals with zeros

    n_add_cut = (sg.round_power_of_2(2, len(q_cut)) - len(q_cut)) // 2  # 'round' to next power of 2
    q_for_nft, t_for_nft = add_lateral_to_signal(q_cut, slope_zero, t_cut, n_add_cut)

    # result_dbp_nft = nft.make_dbp_fnft(q_for_nft, t_for_nft - (t_for_nft[-1] + t_for_nft[0]) / 2, z_prop,
    #                                    xi_upsampling=xi_upsampling, inverse_type='tib', fnft_type=fnft_type,
    #                                    print_sys_message=print_sys_message)

    result_dbp_nft = nft.make_dbp_nft(q, t_for_nft - (t_for_nft[-1] + t_for_nft[0]) / 2, z_prop,
                                      xi_upsampling=xi_upsampling,
                                      forward_continuous_type=forward_continuous_type,
                                      forward_discrete_type=forward_discrete_type,
                                      forward_discrete_coef_type=forward_discrete_coef_type,
                                      inverse_type=inverse_type,
                                      fnft_type=fnft_type, nft_type=nft_type,
                                      use_contour=use_contour, n_discrete_skip=n_discrete_skip,
                                      print_sys_message=print_sys_message)

    if inverse_type == 'both' or inverse_type == 'tib':
        q_nft = result_dbp_nft['q_total']
    elif inverse_type == 'fnft':
        q_nft = result_dbp_nft['q_fnft']
    else:
        print("[process_nft_window]: wrong inverse_type")

    return q_nft, t_for_nft, result_dbp_nft, n_add_cut


def get_points_from_signal(q, np_symb, t_symb, n_car, modulation_type, scale_coef, n_add=0):

    points = np.array(
        sg.get_points_wdm(q, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=modulation_type,
                          n_lateral=n_add))
    points_scaled = scale_coef * points
    points_found = sg.get_nearest_constellation_points(points_scaled, modulation_type)

    return points, points_scaled, points_found


def get_points_range(points, ns_proc, ns_d):
    return points[ns_d:ns_d + ns_proc]


def get_restore_error(q_original, q_restored, points_original, points_restored, n_add=0):

    # calculate MSE
    # for n_add > 0 MSE is calculated for central part of the signal
    np_q_proc = len(q_original) - 2 * n_add
    diff = np.absolute(q_original[n_add: n_add + np_q_proc] - q_restored[n_add: n_add + np_q_proc])
    mse = np.mean(np.power(np.absolute(q_original[n_add: n_add + np_q_proc] - q_restored[n_add: n_add + np_q_proc]), 2))

    max_point_diff = np.max(np.absolute(points_original - points_restored))

    return diff, mse, max_point_diff


def process_signal_fnft(q_original, q_prop, z_prop, np_symb, t_symb, n_car, modulation_type,
                        scale_coef, n_add=0,
                        ns_proc=8, ns_d=130, ns_skip=0, n_steps=1,
                        window_mode='cdc',
                        fnft_type=0, xi_upsampling=4,
                        print_sys_message=False):
    """
    Same procedure as process_signal_nft, but it uses only FNFT.

    Args:
        q_original: original signal from Tx (transmitter)
        q_prop: propagated signal at Rx (receiver)
        z_prop: dimensionless propagation distance
        np_symb: number of points per symbol
        t_symb: dimensionless symbol time interval. Time length of one symbol
        n_car: number of WDM carriers (only n_car=1 works now)
        modulation_type: type of modulation
        scale_coef: scale coefficient for constellation
        n_add: (default = 0) number of additional points on the left and right sides of the signal. See README.md for explanation.
        ns_proc: (default = 8) number of symbols to process
        ns_d: (default = 30) number of symbols on the left and right side of processing interval which contain information about chromatic dispersion
        ns_skip: (default = 0) number of symbols from the left to skip
        n_steps: (default = 1) number of processing steps
        window_mode: (default = 'cdc') type of window to cut the signal

            - 'cdc' -- CDC-Windowed mode. 1. Compensate chromatic dispersion. 2. Cut signal. 3. De-compensate (return) chromatic dispersion. 4. Process signal
            - 'plain' -- cut signal without any manipulation with it.

        fnft_type: type of FNFT calculation, default = 0
        xi_upsampling: (default = 1) upsampling in nonlinear domain. n_xi = xi_upsampling * n_t
        print_sys_message: print additional messages during calculation, default = False

    Returns:
        Dictionary. Points from WDM signal from NFT-DBP and CDC, and MSEs for signals

        - 'points' -- constellation points from NFT-DBP procedure
        - 'points_scaled' -- points multiplied by scale_coef
        - 'points_found' -- closest points in constellation for points_scaled
        - 'points_cdc'  -- constellation points from CDC procedure
        - 'points_cdc_scaled' -- points_cdc multiplied by scale_coef
        - 'points_cdc_found' -- closest points in constellation for points_cdc_scaled
        - 'points_original' -- original constellation points
        - 'points_original_scaled' -- points_original multiplied by scale_coef
        - 'points_original_found' -- closest points in constellation for points_original_scaled
        - 'mse_nft' -- MSE for NFT restored signal and original one. Uses full ns_d + ns_proc + ns_d number of symbols
        - 'mse_nft_range' -- same but uses only processing interval (ns_proc symbols)
        - 'max_point_diff_nft' -- absolute value of maximum distance between found by NFT and original point in constellation
        - 'mse_cdc' -- MSE for CDC restored signal and original one. Uses full ns_d + ns_proc + ns_d number of symbols
        - 'mse_cdc_range' -- same but uses only processing interval (ns_proc symbols)
        - 'max_point_diff_cdc' - absolute value of maximum distance between found by CDC and original point in constellation

    """

    # n_add -- number of points which was added to the signal on the left side of first time slot
    # for burst mode n_add equals to the sum of lateral points (shape of the signal, e.g. sinc) and zeros points

    # parameters of the line
    beta2 = -1.0

    # parameters to signal process
    dt = t_symb / np_symb
    n_t = len(q_prop)
    t_span = (n_t - 1) * dt

    # window_mode = 'cdc'
    # xi_upsampling = 4

    # ns_proc = 8
    # ns_d = 30 - ns_proc // 2
    ns_total = ns_proc + 2 * ns_d
    # ns_skip = 712  # initial number of symbols to skip to eliminate border effects

    step = ns_proc  # step size in number of symbols for signal processing. Each iteration window shifts to step
    # n_steps = 200  # total number of steps (iterations)

    # lists of all points found
    # for NFT
    points = []
    points_scaled = []
    points_found = []
    # for CDC
    points_cdc = []
    points_cdc_scaled = []
    points_cdc_found = []
    # original points
    points_original = []
    points_original_scaled = []
    points_original_found = []

    # lists of all error for each processing step
    # for NFT
    diff_nft = []
    diff_nft_range = []
    mse_nft = []
    mse_nft_range = []
    max_point_diff_nft = []
    # for CDC
    diff_cdc = []
    diff_cdc_range = []
    mse_cdc = []
    mse_cdc_range = []
    max_point_diff_cdc = []

    for proc_iter in range(n_steps):

        # np_skip = ns_skip * np_symb + n_add

        # make nft to restore original signal
        q_nft, t_nft, result_dbp_nft, n_add_nft = process_fnft_window(q_prop, z_prop, np_symb, t_symb, ns_proc, ns_d,
                                                                      ns_skip, n_add, window_mode=window_mode,
                                                                      xi_upsampling=xi_upsampling,
                                                                      fnft_type=fnft_type,
                                                                      print_sys_message=print_sys_message)
        t_points = get_points_from_signal(q_nft, np_symb, t_symb, n_car, modulation_type,
                                          scale_coef, n_add=n_add_nft)

        # make CDC to restore original signal
        w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
        q_cdc = sg.dispersion_compensation_t(q_prop, w, z_prop, beta2)
        # cut CDC-compensated signal
        q_cut_cdc, t_cut_cdc = get_sub_signal(q_cdc, np_symb, t_symb, ns_total, ns_skip, n_add)
        q_cut_cdc_t, t_cut_cdc_t = add_lateral_to_signal(q_cut_cdc, slope_zero, t_cut_cdc, n_add_nft)
        t_points_cdc = get_points_from_signal(q_cut_cdc, np_symb, t_symb, n_car, modulation_type, scale_coef)

        # cut original signal to calculate errors and get points
        q_cut, t_cut = get_sub_signal(q_original, np_symb, t_symb, ns_total, ns_skip, n_add)
        q_cut_t, t_cut_t = add_lateral_to_signal(q_cut, slope_zero, t_cut, n_add_nft)
        t_points_original = get_points_from_signal(q_cut, np_symb, t_symb, n_car, modulation_type, scale_coef)

        # take only processing points
        t_points_range = get_points_range(t_points[0], ns_proc, ns_d)  # points from the signal
        t_points_range_scaled = get_points_range(t_points[1], ns_proc, ns_d)  # scale points
        t_points_range_found = get_points_range(t_points[2], ns_proc, ns_d)  # find nearest constellation points

        t_points_cdc_range = get_points_range(t_points_cdc[0], ns_proc, ns_d)
        t_points_cdc_range_scaled = get_points_range(t_points_cdc[1], ns_proc, ns_d)
        t_points_cdc_range_found = get_points_range(t_points_cdc[2], ns_proc, ns_d)

        t_points_orig_range = get_points_range(t_points_original[0], ns_proc, ns_d)
        t_points_orig_range_scaled = get_points_range(t_points_original[1], ns_proc, ns_d)
        t_points_orig_range_found = get_points_range(t_points_original[2], ns_proc, ns_d)

        # add points to total list of points
        points = np.concatenate((points, t_points_range))
        points_scaled = np.concatenate((points_scaled, t_points_range_scaled))
        points_found = np.concatenate((points_found, t_points_range_found))

        points_cdc = np.concatenate((points_cdc, t_points_cdc_range))
        points_cdc_scaled = np.concatenate((points_cdc_scaled, t_points_cdc_range_scaled))
        points_cdc_found = np.concatenate((points_cdc_found, t_points_cdc_range_found))

        points_original = np.concatenate((points_original, t_points_orig_range))
        points_original_scaled = np.concatenate((points_original_scaled, t_points_orig_range_scaled))
        points_original_found = np.concatenate((points_original_found, t_points_orig_range_found))

        ns_skip += step

        if print_sys_message:
            print("[" + window_mode + "] PER TIB:", sg.get_points_error_rate(t_points_orig_range_found, t_points_range_found),
                  "| BER TIB:", sg.get_ber_by_points(t_points_orig_range_found, t_points_range_found, modulation_type)
                  )

        # calculate errors
        # for NFT
        t_diff_nft, t_mse_nft, t_max_point_diff_nft = get_restore_error(q_cut_t, q_nft, t_points_orig_range,
                                                                        t_points_range, n_add=0)
        t_diff_nft_range, t_mse_nft_range, _ = get_restore_error(q_cut_t, q_nft, t_points_orig_range, t_points_range,
                                                                 n_add=n_add_nft + ns_d * np_symb)

        # for CDC
        t_diff_cdc, t_mse_cdc, t_max_point_diff_cdc = get_restore_error(q_cut_t, q_cut_cdc_t, t_points_orig_range,
                                                                        t_points_cdc_range, n_add=0)
        t_diff_cdc_range, t_mse_cdc_range, _ = get_restore_error(q_cut_t, q_cut_cdc_t, t_points_orig_range,
                                                                 t_points_cdc_range, n_add=n_add_nft + ns_d * np_symb)

        # add errors to total list of errors
        # diff_nft = np.concatenate((diff_nft, t_diff_nft))
        # diff_nft_range = np.concatenate((diff_nft_range, t_diff_nft_range))
        mse_nft = np.concatenate((mse_nft, [t_mse_nft]))
        mse_nft_range = np.concatenate((mse_nft_range, [t_mse_nft_range]))
        max_point_diff_nft = np.concatenate((max_point_diff_nft, [t_max_point_diff_nft]))

        # diff_cdc = np.concatenate((diff_cdc, t_diff_cdc))
        # diff_cdc_range = np.concatenate((diff_cdc_range, t_diff_cdc_range))
        mse_cdc = np.concatenate((mse_cdc, [t_mse_cdc]))
        mse_cdc_range = np.concatenate((mse_cdc_range, [t_mse_cdc_range]))
        max_point_diff_cdc = np.concatenate((max_point_diff_cdc, [t_max_point_diff_cdc]))

    result = {'points': points,
              'points_scaled': points_scaled,
              'points_found': points_found,
              'points_cdc': points_cdc,
              'points_cdc_scaled': points_cdc_scaled,
              'points_cdc_found': points_cdc_found,
              'points_original': points_original,
              'points_original_scaled': points_original_scaled,
              'points_original_found': points_original_found,
              # 'diff_nft': diff_nft,
              # 'diff_nft_range': diff_nft_range,
              'mse_nft': mse_nft,
              'mse_nft_range': mse_nft_range,
              'max_point_diff_nft': max_point_diff_nft,
              # 'diff_cdc': diff_cdc,
              # 'diff_cdc_range': diff_cdc_range,
              'mse_cdc': mse_cdc,
              'mse_cdc_range': mse_cdc_range,
              'max_point_diff_cdc': max_point_diff_cdc
              }

    return result


def process_signal(q_original, q_prop, z_prop, np_symb, t_symb, n_car, modulation_type,
                   scale_coef, n_add=0,
                   ns_proc=8, ns_d=30, ns_skip=0, n_steps=1,
                   window_mode='cdc',
                   xi_upsampling=1,
                   forward_continuous_type='fnft',
                   forward_discrete_type='fnft',
                   forward_discrete_coef_type='fnftpoly',
                   inverse_type='both',
                   fnft_type=0, nft_type='bo',
                   use_contour=False, n_discrete_skip=10,
                   print_sys_message=False):
    """
    Takes signal and makes DBP with NFT. Restore signal, restore points from WDM signal.
    Also, it compares NFT-DBP with CDC. But it doesn't make phase rotation.

    Args:
        q_original: original signal from Tx (transmitter)
        q_prop: propagated signal at Rx (receiver)
        z_prop: dimensionless propagation distance
        np_symb: number of points per symbol
        t_symb: dimensionless symbol time interval. Time length of one symbol
        n_car: number of WDM carriers (only n_car=1 works now)
        modulation_type: type of modulation
        scale_coef: scale coefficient for constellation
        n_add: (default = 0) number of additional points on the left and right sides of the signal. See README.md for explanation.
        ns_proc: (default = 8) number of symbols to process
        ns_d: (default = 30) number of symbols on the left and right side of processing interval which contain information about chromatic dispersion
        ns_skip: (default = 0) number of symbols from the left to skip
        n_steps: (default = 1) number of processing steps
        window_mode: (default = 'cdc') type of window to cut the signal

            - 'cdc' -- CDC-Windowed mode. 1. Compensate chromatic dispersion. 2. Cut signal. 3. De-compensate (return) chromatic dispersion. 4. Process signal
            - 'plain' -- cut signal without any manipulation with it.

        xi_upsampling: (default = 1) upsampling in nonlinear domain. n_xi = xi_upsampling * n_t
        forward_continuous_type: type of calculation of continuous spectrum for forward NFT

            - 'fnft' -- use FNFT to calculate continuous spectrum (real axe only!)
            - 'fnftpoly' -- use polynomials from FNFT to calculate continuous spectrum (arbitrary contour)
            - 'slow' -- use transfer matrices to calculate continuous spectrum (arbitrary contour)

        forward_discrete_type:

            - 'fnft' -- use FNFT to calculate discrete spectrum points (coefficients calculated automatically)
            - 'pjt' -- use PJT (phase jump tracking)
            - 'roots' -- not implemented (other procedures to find polynomial roots -> discrete spectrum points)

        forward_discrete_coef_type:

            - 'fnftpoly' -- use polynomial from FNFT to calculate spectral coefficients (b-coefficient is not stable for eigenvalues with bit imaginary part)
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
        Dictionary. Points from WDM signal from NFT-DBP and CDC, and MSEs for signals

        - 'points' -- constellation points from NFT-DBP procedure
        - 'points_scaled' -- points multiplied by scale_coef
        - 'points_found' -- closest points in constellation for points_scaled
        - 'points_cdc'  -- constellation points from CDC procedure
        - 'points_cdc_scaled' -- points_cdc multiplied by scale_coef
        - 'points_cdc_found' -- closest points in constellation for points_cdc_scaled
        - 'points_original' -- original constellation points
        - 'points_original_scaled' -- points_original multiplied by scale_coef
        - 'points_original_found' -- closest points in constellation for points_original_scaled
        - 'mse_nft' -- MSE for NFT restored signal and original one. Uses full ns_d + ns_proc + ns_d number of symbols
        - 'mse_nft_range' -- same but uses only processing interval (ns_proc symbols)
        - 'max_point_diff_nft' -- absolute value of maximum distance between found by NFT and original point in constellation
        - 'mse_cdc' -- MSE for CDC restored signal and original one. Uses full ns_d + ns_proc + ns_d number of symbols
        - 'mse_cdc_range' -- same but uses only processing interval (ns_proc symbols)
        - 'max_point_diff_cdc' - absolute value of maximum distance between found by CDC and original point in constellation

    """

    # n_add -- number of points which was added to the signal on the left side of first time slot
    # for burst mode n_add equals to the sum of lateral points (shape of the signal, e.g. sinc) and zeros points

    # parameters of the line
    beta2 = -1.0

    # parameters to signal process
    dt = t_symb / np_symb
    n_t = len(q_prop)
    t_span = (n_t - 1) * dt

    # window_mode = 'cdc'
    # xi_upsampling = 4

    # ns_proc = 8
    # ns_d = 30 - ns_proc // 2
    ns_total = ns_proc + 2 * ns_d
    # ns_skip = 712  # initial number of symbols to skip to eliminate border effects

    step = ns_proc  # step size in number of symbols for signal processing. Each iteration window shifts to step
    # n_steps = 200  # total number of steps (iterations)

    # lists of all points found
    # for NFT
    points = []
    points_scaled = []
    points_found = []
    # for CDC
    points_cdc = []
    points_cdc_scaled = []
    points_cdc_found = []
    # original points
    points_original = []
    points_original_scaled = []
    points_original_found = []

    # lists of all error for each processing step
    # for NFT
    diff_nft = []
    diff_nft_range = []
    mse_nft = []
    mse_nft_range = []
    max_point_diff_nft = []
    # for CDC
    diff_cdc = []
    diff_cdc_range = []
    mse_cdc = []
    mse_cdc_range = []
    max_point_diff_cdc = []

    for proc_iter in range(n_steps):

        # np_skip = ns_skip * np_symb + n_add

        # make nft to restore original signal
        q_nft, t_nft, result_dbp_nft, n_add_nft = process_nft_window(q_prop, z_prop, np_symb, t_symb, ns_proc, ns_d,
                                                                     ns_skip, n_add,
                                                                     window_mode=window_mode,
                                                                     xi_upsampling=xi_upsampling,
                                                                     forward_continuous_type=forward_continuous_type,
                                                                     forward_discrete_type=forward_discrete_type,
                                                                     forward_discrete_coef_type=forward_discrete_coef_type,
                                                                     inverse_type=inverse_type,
                                                                     fnft_type=fnft_type, nft_type=nft_type,
                                                                     use_contour=use_contour,
                                                                     n_discrete_skip=n_discrete_skip,
                                                                     print_sys_message=print_sys_message)

        t_points = get_points_from_signal(q_nft, np_symb, t_symb, n_car, modulation_type, scale_coef, n_add=n_add_nft)

        # make CDC to restore original signal
        w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
        q_cdc = sg.dispersion_compensation_t(q_prop, w, z_prop, beta2)
        # cut CDC-compensated signal
        q_cut_cdc, t_cut_cdc = get_sub_signal(q_cdc, np_symb, t_symb, ns_total, ns_skip, n_add)
        q_cut_cdc_t, t_cut_cdc_t = add_lateral_to_signal(q_cut_cdc, slope_zero, t_cut_cdc, n_add_nft)
        t_points_cdc = get_points_from_signal(q_cut_cdc, np_symb, t_symb, n_car, modulation_type, scale_coef)

        # cut original signal to calculate errors and get points
        q_cut, t_cut = get_sub_signal(q_original, np_symb, t_symb, ns_total, ns_skip, n_add)
        q_cut_t, t_cut_t = add_lateral_to_signal(q_cut, slope_zero, t_cut, n_add_nft)
        t_points_original = get_points_from_signal(q_cut, np_symb, t_symb, n_car, modulation_type, scale_coef)

        # take only processing points
        t_points_range = get_points_range(t_points[0], ns_proc, ns_d)  # points from the signal
        t_points_range_scaled = get_points_range(t_points[1], ns_proc, ns_d)  # scale points
        t_points_range_found = get_points_range(t_points[2], ns_proc, ns_d)  # find nearest constellation points

        t_points_cdc_range = get_points_range(t_points_cdc[0], ns_proc, ns_d)
        t_points_cdc_range_scaled = get_points_range(t_points_cdc[1], ns_proc, ns_d)
        t_points_cdc_range_found = get_points_range(t_points_cdc[2], ns_proc, ns_d)

        t_points_orig_range = get_points_range(t_points_original[0], ns_proc, ns_d)
        t_points_orig_range_scaled = get_points_range(t_points_original[1], ns_proc, ns_d)
        t_points_orig_range_found = get_points_range(t_points_original[2], ns_proc, ns_d)

        # add points to total list of points
        points = np.concatenate((points, t_points_range))
        points_scaled = np.concatenate((points_scaled, t_points_range_scaled))
        points_found = np.concatenate((points_found, t_points_range_found))

        points_cdc = np.concatenate((points_cdc, t_points_cdc_range))
        points_cdc_scaled = np.concatenate((points_cdc_scaled, t_points_cdc_range_scaled))
        points_cdc_found = np.concatenate((points_cdc_found, t_points_cdc_range_found))

        points_original = np.concatenate((points_original, t_points_orig_range))
        points_original_scaled = np.concatenate((points_original_scaled, t_points_orig_range_scaled))
        points_original_found = np.concatenate((points_original_found, t_points_orig_range_found))

        ns_skip += step

        if print_sys_message:
            print("[" + window_mode + "] PER TIB:", sg.get_points_error_rate(t_points_orig_range_found, t_points_range_found),
                  "| BER TIB:", sg.get_ber_by_points(t_points_orig_range_found, t_points_range_found, modulation_type)
                  )

        # calculate errors
        # for NFT
        t_diff_nft, t_mse_nft, t_max_point_diff_nft = get_restore_error(q_cut_t, q_nft, t_points_orig_range,
                                                                        t_points_range, n_add=0)
        t_diff_nft_range, t_mse_nft_range, _ = get_restore_error(q_cut_t, q_nft, t_points_orig_range, t_points_range,
                                                                 n_add=n_add_nft + ns_d * np_symb)

        # for CDC
        t_diff_cdc, t_mse_cdc, t_max_point_diff_cdc = get_restore_error(q_cut_t, q_cut_cdc_t, t_points_orig_range,
                                                                        t_points_cdc_range, n_add=0)
        t_diff_cdc_range, t_mse_cdc_range, _ = get_restore_error(q_cut_t, q_cut_cdc_t, t_points_orig_range,
                                                                 t_points_cdc_range, n_add=n_add_nft + ns_d * np_symb)

        # add errors to total list of errors
        # diff_nft = np.concatenate((diff_nft, t_diff_nft))
        # diff_nft_range = np.concatenate((diff_nft_range, t_diff_nft_range))
        mse_nft = np.concatenate((mse_nft, [t_mse_nft]))
        mse_nft_range = np.concatenate((mse_nft_range, [t_mse_nft_range]))
        max_point_diff_nft = np.concatenate((max_point_diff_nft, [t_max_point_diff_nft]))

        # diff_cdc = np.concatenate((diff_cdc, t_diff_cdc))
        # diff_cdc_range = np.concatenate((diff_cdc_range, t_diff_cdc_range))
        mse_cdc = np.concatenate((mse_cdc, [t_mse_cdc]))
        mse_cdc_range = np.concatenate((mse_cdc_range, [t_mse_cdc_range]))
        max_point_diff_cdc = np.concatenate((max_point_diff_cdc, [t_max_point_diff_cdc]))

    result = {'points': points,
              'points_scaled': points_scaled,
              'points_found': points_found,
              'points_cdc': points_cdc,
              'points_cdc_scaled': points_cdc_scaled,
              'points_cdc_found': points_cdc_found,
              'points_original': points_original,
              'points_original_scaled': points_original_scaled,
              'points_original_found': points_original_found,
              # 'diff_nft': diff_nft,
              # 'diff_nft_range': diff_nft_range,
              'mse_nft': mse_nft,
              'mse_nft_range': mse_nft_range,
              'max_point_diff_nft': max_point_diff_nft,
              # 'diff_cdc': diff_cdc,
              # 'diff_cdc_range': diff_cdc_range,
              'mse_cdc': mse_cdc,
              'mse_cdc_range': mse_cdc_range,
              'max_point_diff_cdc': max_point_diff_cdc
              }

    return result




#
#
# def process_init():
#     total_points_original_proc = []
#     total_points_nft_proc = []
#     total_points_nft_pred = []
#     total_points_nft_cdc_proc = []
#     total_points_nft_cdc_pred = []
#
#     for n_z_span_proc in range(7, 13):
#         # choose z_prop
#         z_80km_nd = sg.z_km_to_nd(80, t_symb=14.8)
#         # n_z_span_proc = 12
#         z_prop = n_z_span_proc * z_80km_nd
#         q_prop = q_prop_total[n_z_span_proc - 1]
#         q = q_init
#
#         # beta2 = -1.0
#
#         # make dispersion compensation
#         w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
#         q_cdc = sg.dispersion_compensation_t(q_prop, w, z_prop, beta2)
#
#         points_original_proc = []
#         points_nft_proc = []
#         points_nft_pred = []
#         points_nft_cdc_proc = []
#         points_nft_cdc_pred = []
#
#         num_symb_proc = 8
#         num_symb_d = 400 - num_symb_proc // 2
#         # num_symb_skip_shift = (num_symbols // 2 - num_symb_d - num_symb_proc // 2) -2 # initial shift from the left size
#         num_symb_skip_shift = 756
#
#         step = num_symb_proc  # for how much symb we shift every step
#         # but I will process only half (left) of them
#
#         num_symb_skip_base = num_symb_d + num_symb_skip_shift  # desire skip to the proc interval. Have to be bigger that num_symb_d
#         num_symb_total = 2 * num_symb_d + num_symb_proc  # total number of symbols to cut
#
#         # n_steps_for_cut = (num_symbols - 2 * num_symb_skip_shift) // step
#         n_steps_for_cut = 8
#         for proc_iter in range(n_steps_for_cut):
#             # proc_iter = 0
#             print('-----^_^-----', 'iteration', proc_iter + 1, 'of', n_steps_for_cut, '-----^_^-----')
#             # calculate shift in time domain
#             num_symb_skip = num_symb_skip_base - num_symb_d + proc_iter * step
#             # num_symb_skip = 0
#
#             # Cut part of the signal to process with FNFT
#             signal_cut, t_cut = get_sub_signal(q_prop, np_symb, t_symb, num_symb_total, num_symb_skip,
#                                                n_add + n_lateral)
#             n_add_cut = (sg.round_power_of_2(1, len(signal_cut)) - len(signal_cut)) // 2
#             signal_proc, t_proc = add_lateral_to_signal(signal_cut, slope, t_cut, n_add_cut)
#
#             q_cdc_cut, t_cdc_cut = get_sub_signal(q_cdc, np_symb, t_symb, num_symb_total, num_symb_skip,
#                                                   n_add + n_lateral)
#             q_cdc_proc, t_cdc_proc = add_lateral_to_signal(q_cdc_cut, slope, t_cdc_cut, n_add_cut)
#
#             # cut cdc-compensated signal and inverse cdc (return to propagated signal)
#             n_symb_cdc_add = 35 * n_z_span_proc  # 400 for 960 km
#             if n_symb_cdc_add < 128:
#                 n_symb_cdc_add = 128
#             n_add_cut_cdc = n_symb_cdc_add * np_symb
#             q_cdc_window = cut_signal_window(q_cdc, np_symb, t_symb, num_symb_total, num_symb_skip, n_add + n_lateral)
#             w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
#             q_cdc_prop = sg.dispersion_compensation_t(q_cdc_window, w, -z_prop, beta2)
#             q_cdc_for_nft, t_cdc_for_nft = get_sub_signal(q_cdc_prop, np_symb, t_symb,
#                                                           num_symb_total + 2 * n_symb_cdc_add,
#                                                           num_symb_skip - n_symb_cdc_add, n_add + n_lateral)
#
#             # get init signal in the same time window
#
#             q_init_cut, t_init_cut = get_sub_signal(q, np_symb, t_symb, num_symb_total, num_symb_skip,
#                                                     n_add + n_lateral)
#             q_init_proc, t_init_proc = add_lateral_to_signal(q_init_cut, slope, t_init_cut, n_add_cut)
#
#             points_rc = np.array(
#                 sg.get_points_wdm(q_init_proc, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=mod_type,
#                                   n_lateral=n_add_cut))
#             scale_coef = sg.get_scale_coef(points_rc, mod_type)
#             points_rc_test = sg.get_nearest_constellation_points(scale_coef * points_rc, mod_type)
#
#             # print(np.max(points_rc_test - points_orig))
#             # num_symb_proc_cut = num_symb_proc // 2
#             num_symb_proc_cut = 0
#             final_points_test = points_rc_test[num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]
#             final_points_orig = points_orig[
#                                 num_symb_skip + num_symb_d: num_symb_skip + num_symb_d + num_symb_proc - num_symb_proc_cut]
#             points_original_proc = np.concatenate((points_original_proc, final_points_orig))
#             final_diff = np.max(final_points_orig - final_points_test)
#             # print(len(points_rc_test), final_points_test)
#             # print(len(points_orig), final_points_orig)
#             # print(final_diff)
#             if sg.get_points_error_rate(final_points_orig, final_points_test)[1] != 0:
#                 print("ERROR: initial points are not the same!")
#                 print("PER w/o propagation:", sg.get_points_error_rate(final_points_orig, final_points_test),
#                       "| BER w/o propagation:", sg.get_ber_by_points(final_points_orig, final_points_test, mod_type),
#                       "| {----- have to be 0 -----}")
#
#             # make dbp with nft
#             # check skip and comment -----vvvvv-----
#             xi_upsampling = 4
#             # result_dbp_nft = nft.make_dbp_nft(signal_proc, t_proc - (t_proc[-1] + t_proc[0]) / 2, z_prop,
#             #                                   xi_upsampling=xi_upsampling, inverse_type='tib', print_sys_message=True)
#             # q_total = result_dbp_nft['q_total']
#             # q_fnft = result_dbp_nft['q_fnft']
#
#             # to skip dbp with nft -----vvvvv-----
#             q_total = signal_proc
#
#             # make dbp with nft-cdc
#             test_coef_cdc = 1  # not used
#             # xi_upsampling = 8
#             result_dbp_nft_cdc = nft.make_dbp_nft(q_cdc_for_nft / test_coef_cdc,
#                                                   t_cdc_for_nft - (t_cdc_for_nft[-1] + t_cdc_for_nft[0]) / 2, z_prop,
#                                                   xi_upsampling=xi_upsampling, inverse_type='tib',
#                                                   print_sys_message=True)
#             q_total_for_cdc = result_dbp_nft_cdc['q_total'] * test_coef_cdc
#             # q_fnft_for_cdc = result_dbp_nft_cdc['q_fnft']
#
#             # calculate NMSE
#             n_q_proc = len(q_init_proc) - 2 * n_add_cut
#             diff = np.absolute(q_init_proc[n_add_cut: n_add_cut + n_q_proc] - q_total[n_add_cut: n_add_cut + n_q_proc])
#
#             ind_start = n_add_cut + num_symb_d * np_symb
#             ind_end = n_add_cut + num_symb_d * np_symb + (num_symb_proc - num_symb_proc_cut) * np_symb
#             diff_process = np.absolute(q_init_proc[ind_start: ind_end] - q_total[ind_start: ind_end])
#             # print("[pure] full interval mean/max:", np.mean(diff), np.max(diff),
#             #       "| [pure] process interval mean/max:", np.mean(diff_process), np.max(diff_process))
#
#             # for cdc
#
#             n_q_proc_cdc = len(q_total_for_cdc) - 2 * n_add_cut_cdc
#             diff = np.absolute(q_init_proc[n_add_cut: n_add_cut + n_q_proc] - q_total_for_cdc[
#                                                                               n_add_cut_cdc: n_add_cut_cdc + n_q_proc_cdc])
#
#             ind_start_cdc = n_add_cut_cdc + num_symb_d * np_symb
#             ind_end_cdc = n_add_cut_cdc + num_symb_d * np_symb + (num_symb_proc - num_symb_proc_cut) * np_symb
#             diff_process = np.absolute(q_init_proc[ind_start: ind_end] - q_total_for_cdc[ind_start_cdc: ind_end_cdc])
#             # print("[cdc] full interval mean/max:", np.mean(diff), np.max(diff),
#             #       "| [cdc] process interval mean/max:", np.mean(diff_process), np.max(diff_process))
#
#             # restore points
#
#             points_tib = np.array(
#                 sg.get_points_wdm(q_total, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=mod_type,
#                                   n_lateral=n_add_cut))
#             points_tib_to_scale = scale_coef * points_tib
#             points_tib_found = sg.get_nearest_constellation_points(points_tib_to_scale, mod_type)
#
#             # print(np.max(points_rc_test - points_orig))
#
#             final_points_tib = points_tib_found[num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]
#             points_nft_proc = np.concatenate((points_nft_proc, final_points_tib))
#             final_diff = np.max(final_points_orig - final_points_tib)
#             # print(len(points_tib), len(final_points_tib), final_points_tib)
#             # print(len(points_orig), len(final_points_orig), final_points_orig)
#             # print(final_diff)
#             points_nft_pred = np.concatenate(
#                 (points_nft_pred, points_tib_to_scale[num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]))
#
#             # restore points cdc nft
#
#             points_tib_for_cdc = np.array(
#                 sg.get_points_wdm(q_total_for_cdc, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=mod_type,
#                                   n_lateral=n_add_cut_cdc))
#             points_tib_to_scale_for_cdc = scale_coef * points_tib_for_cdc
#             points_tib_found_for_cdc = sg.get_nearest_constellation_points(points_tib_to_scale_for_cdc, mod_type)
#
#             # print(np.max(points_rc_test - points_orig))
#
#             final_points_tib_for_cdc = points_tib_found_for_cdc[
#                                        num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]
#             points_nft_cdc_proc = np.concatenate((points_nft_cdc_proc, final_points_tib_for_cdc))
#             final_diff_for_cdc = np.max(final_points_orig - final_points_tib_for_cdc)
#             # print(len(points_tib_for_cdc), len(final_points_tib_for_cdc), final_points_tib_for_cdc)
#             # print(len(points_orig), len(final_points_orig), final_points_orig)
#             # print(final_diff_for_cdc)
#             points_nft_cdc_pred = np.concatenate((points_nft_cdc_pred, points_tib_to_scale_for_cdc[
#                                                                        num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]))
#
#             print("[pure / cdc] PER TIB:", sg.get_points_error_rate(final_points_orig, final_points_tib),
#                   sg.get_points_error_rate(final_points_orig, final_points_tib_for_cdc),
#                   "| BER TIB:", sg.get_ber_by_points(final_points_orig, final_points_tib, mod_type),
#                   sg.get_ber_by_points(final_points_orig, final_points_tib_for_cdc, mod_type))
#
#         total_points_original_proc.append(points_original_proc)
#         total_points_nft_proc.append(points_nft_proc)
#         total_points_nft_pred.append(points_nft_pred)
#         total_points_nft_cdc_proc.append(points_nft_cdc_proc)
#         total_points_nft_cdc_pred.append(points_nft_cdc_pred)
