import numpy as np
import signal_generation as sg
# import nft_analyse as nft

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


def slope(x, param):
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


def cut(q, np_symb, ns_proc, ns_d, ns_skip, n_add = 0, t_symb = 1.):
    # q -- signal to cut
    # np_symb -- number of points in one symbol
    # t_symb -- dimensionless time per one symbol (default = 1)
    # n_add -- additional number of points from the beginning (default = 0)

    ns_total = 2 * ns_d + ns_proc  # total number of symbols for processing

    # Cut part of the signal to process with FNFT
    signal_cut, t_cut = get_sub_signal(q, np_symb, t_symb, ns_total, ns_skip, n_add)
    n_add_cut = (sg.round_power_of_2(1, len(signal_cut)) - len(signal_cut)) // 2
    signal_proc, t_proc = add_lateral_to_signal(signal_cut, slope, t_cut, n_add_cut)

def process():
    total_points_original_proc = []
    total_points_nft_proc = []
    total_points_nft_pred = []
    total_points_nft_cdc_proc = []
    total_points_nft_cdc_pred = []

    for n_z_span_proc in range(7, 13):
        # choose z_prop
        z_80km_nd = sg.z_km_to_nd(80, t_symb=14.8)
        # n_z_span_proc = 12
        z_prop = n_z_span_proc * z_80km_nd
        q_prop = q_prop_total[n_z_span_proc - 1]
        q = q_init

        # beta2 = -1.0

        # make dispersion compensation
        w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
        q_cdc = sg.dispersion_compensation_t(q_prop, w, z_prop, beta2)

        points_original_proc = []
        points_nft_proc = []
        points_nft_pred = []
        points_nft_cdc_proc = []
        points_nft_cdc_pred = []

        num_symb_proc = 8
        num_symb_d = 400 - num_symb_proc // 2
        # num_symb_skip_shift = (num_symbols // 2 - num_symb_d - num_symb_proc // 2) -2 # initial shift from the left size
        num_symb_skip_shift = 756

        step = num_symb_proc  # for how much symb we shift every step
        # but I will process only half (left) of them

        num_symb_skip_base = num_symb_d + num_symb_skip_shift  # desire skip to the proc interval. Have to be bigger that num_symb_d
        num_symb_total = 2 * num_symb_d + num_symb_proc  # total number of symbols to cut

        # n_steps_for_cut = (num_symbols - 2 * num_symb_skip_shift) // step
        n_steps_for_cut = 8
        for proc_iter in range(n_steps_for_cut):
            # proc_iter = 0
            print('-----^_^-----', 'iteration', proc_iter + 1, 'of', n_steps_for_cut, '-----^_^-----')
            # calculate shift in time domain
            num_symb_skip = num_symb_skip_base - num_symb_d + proc_iter * step
            # num_symb_skip = 0

            # Cut part of the signal to process with FNFT
            signal_cut, t_cut = get_sub_signal(q_prop, np_symb, t_symb, num_symb_total, num_symb_skip,
                                               n_add + n_lateral)
            n_add_cut = (sg.round_power_of_2(1, len(signal_cut)) - len(signal_cut)) // 2
            signal_proc, t_proc = add_lateral_to_signal(signal_cut, slope, t_cut, n_add_cut)

            q_cdc_cut, t_cdc_cut = get_sub_signal(q_cdc, np_symb, t_symb, num_symb_total, num_symb_skip,
                                                  n_add + n_lateral)
            q_cdc_proc, t_cdc_proc = add_lateral_to_signal(q_cdc_cut, slope, t_cdc_cut, n_add_cut)

            # cut cdc-compensated signal and inverse cdc (return to propagated signal)
            n_symb_cdc_add = 35 * n_z_span_proc  # 400 for 960 km
            if n_symb_cdc_add < 128:
                n_symb_cdc_add = 128
            n_add_cut_cdc = n_symb_cdc_add * np_symb
            q_cdc_window = cut_signal_window(q_cdc, np_symb, t_symb, num_symb_total, num_symb_skip, n_add + n_lateral)
            w = fftshift([(i - n_t / 2) * (2. * np.pi / t_span) for i in range(n_t)])
            q_cdc_prop = sg.dispersion_compensation_t(q_cdc_window, w, -z_prop, beta2)
            q_cdc_for_nft, t_cdc_for_nft = get_sub_signal(q_cdc_prop, np_symb, t_symb,
                                                          num_symb_total + 2 * n_symb_cdc_add,
                                                          num_symb_skip - n_symb_cdc_add, n_add + n_lateral)

            # get init signal in the same time window

            q_init_cut, t_init_cut = get_sub_signal(q, np_symb, t_symb, num_symb_total, num_symb_skip,
                                                    n_add + n_lateral)
            q_init_proc, t_init_proc = add_lateral_to_signal(q_init_cut, slope, t_init_cut, n_add_cut)

            points_rc = np.array(
                sg.get_points_wdm(q_init_proc, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=mod_type,
                                  n_lateral=n_add_cut))
            scale_coef = sg.get_scale_coef(points_rc, mod_type)
            points_rc_test = sg.get_nearest_constellation_points(scale_coef * points_rc, mod_type)

            # print(np.max(points_rc_test - points_orig))
            # num_symb_proc_cut = num_symb_proc // 2
            num_symb_proc_cut = 0
            final_points_test = points_rc_test[num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]
            final_points_orig = points_orig[
                                num_symb_skip + num_symb_d: num_symb_skip + num_symb_d + num_symb_proc - num_symb_proc_cut]
            points_original_proc = np.concatenate((points_original_proc, final_points_orig))
            final_diff = np.max(final_points_orig - final_points_test)
            # print(len(points_rc_test), final_points_test)
            # print(len(points_orig), final_points_orig)
            # print(final_diff)
            if sg.get_points_error_rate(final_points_orig, final_points_test)[1] != 0:
                print("ERROR: initial points are not the same!")
                print("PER w/o propagation:", sg.get_points_error_rate(final_points_orig, final_points_test),
                      "| BER w/o propagation:", sg.get_ber_by_points(final_points_orig, final_points_test, mod_type),
                      "| {----- have to be 0 -----}")

            # make dbp with nft
            # check skip and comment -----vvvvv-----
            xi_upsampling = 4
            # result_dbp_nft = nft.make_dbp_nft(signal_proc, t_proc - (t_proc[-1] + t_proc[0]) / 2, z_prop,
            #                                   xi_upsampling=xi_upsampling, inverse_type='tib', print_sys_message=True)
            # q_total = result_dbp_nft['q_total']
            # q_fnft = result_dbp_nft['q_fnft']

            # to skip dbp with nft -----vvvvv-----
            q_total = signal_proc

            # make dbp with nft-cdc
            test_coef_cdc = 1  # not used
            # xi_upsampling = 8
            result_dbp_nft_cdc = nft.make_dbp_nft(q_cdc_for_nft / test_coef_cdc,
                                                  t_cdc_for_nft - (t_cdc_for_nft[-1] + t_cdc_for_nft[0]) / 2, z_prop,
                                                  xi_upsampling=xi_upsampling, inverse_type='tib',
                                                  print_sys_message=True)
            q_total_for_cdc = result_dbp_nft_cdc['q_total'] * test_coef_cdc
            # q_fnft_for_cdc = result_dbp_nft_cdc['q_fnft']

            # calculate NMSE
            n_q_proc = len(q_init_proc) - 2 * n_add_cut
            diff = np.absolute(q_init_proc[n_add_cut: n_add_cut + n_q_proc] - q_total[n_add_cut: n_add_cut + n_q_proc])

            ind_start = n_add_cut + num_symb_d * np_symb
            ind_end = n_add_cut + num_symb_d * np_symb + (num_symb_proc - num_symb_proc_cut) * np_symb
            diff_process = np.absolute(q_init_proc[ind_start: ind_end] - q_total[ind_start: ind_end])
            # print("[pure] full interval mean/max:", np.mean(diff), np.max(diff),
            #       "| [pure] process interval mean/max:", np.mean(diff_process), np.max(diff_process))

            # for cdc

            n_q_proc_cdc = len(q_total_for_cdc) - 2 * n_add_cut_cdc
            diff = np.absolute(q_init_proc[n_add_cut: n_add_cut + n_q_proc] - q_total_for_cdc[
                                                                              n_add_cut_cdc: n_add_cut_cdc + n_q_proc_cdc])

            ind_start_cdc = n_add_cut_cdc + num_symb_d * np_symb
            ind_end_cdc = n_add_cut_cdc + num_symb_d * np_symb + (num_symb_proc - num_symb_proc_cut) * np_symb
            diff_process = np.absolute(q_init_proc[ind_start: ind_end] - q_total_for_cdc[ind_start_cdc: ind_end_cdc])
            # print("[cdc] full interval mean/max:", np.mean(diff), np.max(diff),
            #       "| [cdc] process interval mean/max:", np.mean(diff_process), np.max(diff_process))

            # restore points

            points_tib = np.array(
                sg.get_points_wdm(q_total, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=mod_type,
                                  n_lateral=n_add_cut))
            points_tib_to_scale = scale_coef * points_tib
            points_tib_found = sg.get_nearest_constellation_points(points_tib_to_scale, mod_type)

            # print(np.max(points_rc_test - points_orig))

            final_points_tib = points_tib_found[num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]
            points_nft_proc = np.concatenate((points_nft_proc, final_points_tib))
            final_diff = np.max(final_points_orig - final_points_tib)
            # print(len(points_tib), len(final_points_tib), final_points_tib)
            # print(len(points_orig), len(final_points_orig), final_points_orig)
            # print(final_diff)
            points_nft_pred = np.concatenate(
                (points_nft_pred, points_tib_to_scale[num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]))

            # restore points cdc nft

            points_tib_for_cdc = np.array(
                sg.get_points_wdm(q_total_for_cdc, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=mod_type,
                                  n_lateral=n_add_cut_cdc))
            points_tib_to_scale_for_cdc = scale_coef * points_tib_for_cdc
            points_tib_found_for_cdc = sg.get_nearest_constellation_points(points_tib_to_scale_for_cdc, mod_type)

            # print(np.max(points_rc_test - points_orig))

            final_points_tib_for_cdc = points_tib_found_for_cdc[
                                       num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]
            points_nft_cdc_proc = np.concatenate((points_nft_cdc_proc, final_points_tib_for_cdc))
            final_diff_for_cdc = np.max(final_points_orig - final_points_tib_for_cdc)
            # print(len(points_tib_for_cdc), len(final_points_tib_for_cdc), final_points_tib_for_cdc)
            # print(len(points_orig), len(final_points_orig), final_points_orig)
            # print(final_diff_for_cdc)
            points_nft_cdc_pred = np.concatenate((points_nft_cdc_pred, points_tib_to_scale_for_cdc[
                                                                       num_symb_d:num_symb_d + num_symb_proc - num_symb_proc_cut]))

            print("[pure / cdc] PER TIB:", sg.get_points_error_rate(final_points_orig, final_points_tib),
                  sg.get_points_error_rate(final_points_orig, final_points_tib_for_cdc),
                  "| BER TIB:", sg.get_ber_by_points(final_points_orig, final_points_tib, mod_type),
                  sg.get_ber_by_points(final_points_orig, final_points_tib_for_cdc, mod_type))

        total_points_original_proc.append(points_original_proc)
        total_points_nft_proc.append(points_nft_proc)
        total_points_nft_pred.append(points_nft_pred)
        total_points_nft_cdc_proc.append(points_nft_cdc_proc)
        total_points_nft_cdc_pred.append(points_nft_cdc_pred)
