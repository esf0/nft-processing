import numpy as np
import random
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib
from ssfm import ssfm_dispersive_step


# sinc(t) = sin(pi * t) / (pi * t)

def rcos_spec(f, args):
    # args[0] - symbol time
    # args[1] - roll-off factor rho
    t_symb = args[0]
    p = args[1]

    if abs(f) <= (1. - p) / (2. * t_symb):
        return t_symb
    elif (1. - p) / (2. * t_symb) < abs(f) < (1. + p) / (2. * t_symb):
        return t_symb * np.power(np.cos(np.pi * t_symb / (2. * p) * (abs(f) - (1. - p) / (2. * t_symb))), 2)
    else:
        return 0


def rcos(t, args):
    # args[0] - symbol time
    # args[1] - roll-off factor rho
    t_symb = args[0]
    p = args[1]
    if t == 0:
        return 1.
    elif t % t_symb == 0:
        return 0
    else:
        return np.sinc(t / t_symb) * np.cos(np.pi * p * t / t_symb) / (1. - 4. * (p * t / t_symb) ** 2)


def srrcos_spec(f, args):
    # args[0] - symbol time
    # args[1] - roll-off factor rho
    t_symb = args[0]
    p = args[1]

    return np.sqrt(rcos_spec(f, t_symb, p))


def srrcos(t, args):
    # args[0] - symbol time
    # args[1] - roll-off factor rho
    t_symb = args[0]
    p = args[1]
    pi = np.pi

    if t == 0:
        return 1. / t_symb * (1. + p * (4. / pi - 1))
    elif abs(t) == t_symb / (4. * p):
        return p / (t_symb * np.sqrt(2)) * ((1. + 2. / pi) * np.sin(pi / (4. * p)) + (1. - 2. / pi) * np.cos(pi / (4. * p)))
    else:
        return (np.sin(pi * t / t_symb * (1 - p)) + 4. * p * t / t_symb * np.cos(pi * t / t_symb * (1 + p))) / (pi * t * (1. - (4. * p * t / t_symb) ** 2))




def get_constellation_point(bit_data, type="qpsk"):
    # 64QAM -> 6 bits
    # 0b000000 for example
    # 1 bit -> the sigh of the real part: 0 -> '-'; 1 -> '+'
    # 2 and 3 bits -> value to convert to int and multiply by 2 and add 1
    # 00 -> 0 -> 0 * 2 + 1 = 1; 01 -> 1 -> 1 * 2 + 1 = 3;
    # 10 -> 2 -> 2 * 2 + 1 = 5; 11 -> 3 -> 3 * 2 + 1 = 7;
    # 4 bit -> the sigh of the imag part: 0 -> '-'; 1 -> '+'
    # 5 and 6 bits -> same as for real

    n_mod_type = {"qpsk": 1, "16qam": 2, "64qam": 3, "256qam": 4, "1024qam": 5}
    n = n_mod_type[type]

    # if type == "qpsk":
    #     n = 1
    # elif type == "16qam":
    #     n = 2
    # elif type == "64qam":
    #     n = 3
    # elif type == "256qam":
    #     n = 4
    # elif type == "1024qam":
    #     n = 5
    # else:
    #     print("[get_constellation_point]: unknown constellation type")

    # if bit sequence has less number of bit than we need to the constellation type add bits to the beginning
    if len(bit_data) < 2 * n:
        # temp to not change initial data
        temp_bit_data = ''.join('0' for add_bit in range(2 * n - len(bit_data))) + bit_data
    elif len(bit_data) > 2 * n:
        # if length of the sequence has not integer number of subsequence
        # add '0' bits to the beginning
        if len(bit_data) % (2 * n) != 0:
            temp_bit_data = ''.join('0' for add_bit in range(2 * n - (len(bit_data) % (2 * n)))) + bit_data
        else:
            temp_bit_data = bit_data

        # use recursion for subsequences
        points = [get_constellation_point(temp_bit_data[k * 2 * n:(k + 1) * 2 * n], type=type)
                  for k in range(len(temp_bit_data) // (2 * n))]
        # print("[get_constellation_point]: more bits than needed")
        return points
    else:
        temp_bit_data = bit_data

    # generate constellation according to the Gray code (only 1 bit changes for neighbours)
    point = 1 + 1j
    if temp_bit_data[0] == '0':
        point = complex(-1, point.imag)
    if temp_bit_data[n] == '0':
        point = complex(point.real, -1)

    if type != "qpsk":
        point = complex(point.real * (int(temp_bit_data[1:n], 2) * 2 + 1),
                        point.imag * (int(temp_bit_data[n + 1:], 2) * 2 + 1))

    return point


def get_bits_from_constellation_point(point, type="qpsk"):
    # 64QAM -> 6 bits
    # 0b000000 for example
    # 1 bit -> the sigh of the real part: 0 -> '-'; 1 -> '+'
    # 2 and 3 bits -> value to convert to int and multiply by 2 and add 1
    # 00 -> 0 -> 0 * 2 + 1 = 1; 01 -> 1 -> 1 * 2 + 1 = 3;
    # 10 -> 2 -> 2 * 2 + 1 = 5; 11 -> 3 -> 3 * 2 + 1 = 7;
    # 4 bit -> the sigh of the imag part: 0 -> '-'; 1 -> '+'
    # 5 and 6 bits -> same as for real

    n_mod_type = {"qpsk": 1, "16qam": 2, "64qam": 3, "256qam": 4, "1024qam": 5}
    n = n_mod_type[type]

    bit_data = ''
    data_real = ''
    data_imag = ''

    if type != "qpsk":
        data_real = "{0:b}".format(int((np.absolute(point.real) - 1) / 2))
        if len(data_real) < n - 1:
            data_real = ''.join('0' for add_bit in range(n - 1 - len(data_real))) + data_real

        data_imag = "{0:b}".format(int((np.absolute(point.imag) - 1) / 2))
        if len(data_imag) < n - 1:
            data_imag = ''.join('0' for add_bit in range(n - 1 - len(data_imag))) + data_imag

    # real part
    if np.sign(point.real) == 1:
        bit_data = bit_data + ''.join('1')
    else:
        bit_data = bit_data + ''.join('0')

    bit_data = bit_data + data_real

    # imag part
    if np.sign(point.imag) == 1:
        bit_data = bit_data + ''.join('1')
    else:
        bit_data = bit_data + ''.join('0')

    bit_data = bit_data + data_imag

    return bit_data


def get_bits_from_constellation_points(points, type="qpsk"):
    bit_sequence = ''
    for i in range(len(points)):
        symbol_bits = get_bits_from_constellation_point(points[i], type)
        bit_sequence = bit_sequence + symbol_bits

    return bit_sequence


def get_n_bits(type):
    n_bits = {"qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8, "1024qam": 10}
    return n_bits[type]


# generate wdm signal
def get_wdm_symbol(data, t_symb, n_symb, func, func_args, n_carriers=1, mod_type="qpsk", t_lateral=0, n_lateral=0):
    n_bits = get_n_bits(mod_type)

    # if len(data) > (n_bits[mod_type] * n_carriers):
    #     print("[get_wdm_symbol]: length of data more than can be used!")
    # elif len(data) < (n_bits[mod_type] * n_carriers):

    if len(data) != (n_bits * n_carriers):
        print("[get_wdm_symbol]: wrong length of data")
        return 0

    if t_lateral != 0 and n_lateral != 0:
        print("[get_wdm_symbol]: t_lateral set to 0, because n_lateral is not 0")
        t_lateral = 0

    # dt = t_symb / (n_symb - 1)
    dt = t_symb / n_symb

    if t_lateral != 0:
        n_lateral = round(t_lateral / dt)

    n_total = n_symb + 2 * n_lateral
    symbol = np.zeros(n_total, dtype=complex)
    # t = np.array([(i - (n_total - 1) / 2) * dt for i in range(n_total)])
    t = np.array([(i - n_total / 2) * dt for i in range(n_total)])

    # add carriers
    for i_car in range(n_carriers):
        data_carrier = data[i_car * n_bits: (i_car + 1) * n_bits]
        point = get_constellation_point(data_carrier, mod_type)
        symbol += point * np.array([func(i, func_args) for i in t])

    return symbol

def get_wdm_symbol_by_points(points, t_symb, n_symb, func, func_args, n_carriers=1, t_lateral=0, n_lateral=0):

    if len(points) != n_carriers:
        print("[get_wdm_symbol]: wrong length of data")
        return 0

    if t_lateral != 0 and n_lateral != 0:
        print("[get_wdm_symbol_by_point]: t_lateral set to 0, because n_lateral is not 0")
        t_lateral = 0

    # dt = t_symb / (n_symb - 1)
    dt = t_symb / n_symb

    if t_lateral != 0:
        n_lateral = round(t_lateral / dt)

    n_total = n_symb + 2 * n_lateral
    symbol = np.zeros(n_total, dtype=complex)
    # t = np.array([(i - (n_total - 1) / 2) * dt for i in range(n_total)])
    t = np.array([(i - n_total / 2) * dt for i in range(n_total)])

    # add carriers
    for i_car in range(n_carriers):
        symbol += points[i_car] * np.array([func(i, func_args) for i in t])

    return symbol


# generate wdm signal
def get_wdm_signal(data, t_symb, n_symb, func, func_args, n_carriers=1, mod_type="qpsk", t_lateral=0, n_lateral=0,
                   get_symbol=-1):
    n_bits = get_n_bits(mod_type)

    residue = len(data) % (n_carriers * n_bits)
    if residue != 0:
        temp_data = ''.join('0' for add_bit in range(n_bits * n_carriers - residue)) + data
    else:
        temp_data = data

    symbols_number = len(temp_data) // (n_carriers * n_bits)

    # lateral is additional points around symbol interval (left and right sides, symmetrically)
    signal = np.zeros(symbols_number * n_symb + 2 * n_lateral, dtype=complex)
    if get_symbol >= 0:
        one_symbol = np.zeros(symbols_number * n_symb + 2 * n_lateral, dtype=complex)

    for k in range(symbols_number):

        data_sample = data[k * n_bits: (k + 1) * n_bits]
        symbol = get_wdm_symbol(data_sample, t_symb, n_symb, func, func_args, n_carriers=n_carriers, mod_type=mod_type,
                                t_lateral=t_lateral, n_lateral=n_lateral)

        # check len of symbol
        if len(symbol) != (n_symb + 2 * n_lateral):
            print("[get_wdm_signal]: length of the symbol is incorrect")

        for i_symbol in range(n_symb + 2 * n_lateral):
            signal[k * n_symb + i_symbol] += symbol[i_symbol]
            if get_symbol == k:
                one_symbol[k * n_symb + i_symbol] += symbol[i_symbol]

    if get_symbol >= 0:
        return signal, one_symbol
    else:
        return signal

def get_wdm_signal_by_points(points, t_symb, n_symb, func, func_args, n_carriers=1, t_lateral=0, n_lateral=0,
                             get_symbol=-1):

    if len(points) % n_carriers != 0:
        print("[get_wdm_signal_by_points]: wrong number of points")
        return 0

    symbols_number = len(points) // n_carriers

    # lateral is additional points around symbol interval (left and right sides, symmetrically)
    signal = np.zeros(symbols_number * n_symb + 2 * n_lateral, dtype=complex)
    if get_symbol >= 0:
        one_symbol = np.zeros(symbols_number * n_symb + 2 * n_lateral, dtype=complex)

    for k in range(symbols_number):

        symbol_points = points[k * n_carriers: (k + 1) * n_carriers]
        symbol = get_wdm_symbol_by_points(symbol_points, t_symb, n_symb, func, func_args, n_carriers=n_carriers,
                                          t_lateral=t_lateral, n_lateral=n_lateral)

        # check len of symbol
        if len(symbol) != (n_symb + 2 * n_lateral):
            print("[get_wdm_signal]: length of the symbol is incorrect")

        for i_symbol in range(n_symb + 2 * n_lateral):
            signal[k * n_symb + i_symbol] += symbol[i_symbol]
            if get_symbol == k:
                one_symbol[k * n_symb + i_symbol] += symbol[i_symbol]

    if get_symbol >= 0:
        return signal, one_symbol
    else:
        return signal


def get_energy(signal, dt):
    return np.sum(np.power(np.absolute(signal), 2)) * dt


def set_energy(signal, energy, dt):
    return signal * np.sqrt(energy / get_energy(signal, dt))


def get_l1(signal, dt):
    return np.sum(np.absolute(signal)) * dt


def get_average_power(signal, dt):
    return get_energy(signal, dt) / (len(signal) * dt)


def get_average_power_range(signal, dt, n_points_center):

    n_signal = len(signal)
    if n_points_center <= 0:
        n_points_center = n_signal // 2
    low = n_signal // 2 - n_points_center
    if low < 0:
        low = 0
    up = n_signal // 2 + n_points_center
    if up > n_signal - 1:
        up = n_signal - 1
    return get_average_power(signal[low:up], dt)


def set_average_power(signal, dt, power, n_points_center=0):
    if power < 0:
        print("Error: set positive value of power in dimensionless units")
        p_init = 1

    if n_points_center == 0:
        p_init = get_average_power(signal, dt)
    else:
        p_init = get_average_power_range(signal, dt, n_points_center)
    new_signal = np.sqrt(power / p_init) * signal
    if np.absolute(get_average_power(new_signal, dt) - power > 1e-15):
        print(get_average_power(new_signal, dt))
    return new_signal


# detector part

# cut spectrum with detector bandwidth
def cut_spectrum(spectrum, freq, bandwidth):
    if len(freq) != len(spectrum):
        print("Error: spectrum and frequency arrays have different length")
        return -1

    spec_rest = np.zeros(len(freq), dtype=np.complex128)
    spec_cut = np.zeros(len(freq), dtype=np.complex128)
    for i in range(len(freq)):
        if abs(freq[i]) <= bandwidth / 2:
            spec_rest[i] = spectrum[i]
        else:
            spec_cut[i] = spectrum[i]

    return spec_rest, spec_cut


# compensate dispersion in f space
def dispersion_compensation_f(spectrum, freq, z, beta2, beta3=0.0):
    compensated_spectrum = spectrum * np.exp((0.5j * beta2 * freq ** 2 + 1. / 6. * beta3 * freq ** 3) * -1.0 * z)
    return compensated_spectrum


# compensate dispersion in t space
def dispersion_compensation_t(signal, freq, z, beta2, beta3=0.0):
    return ifft(dispersion_compensation_f(fft(signal), freq, z, beta2, beta3))


# get constellation points from wdm
def get_points_wdm_bad(signal, t_symb, n_symb, func, func_args, n_carriers=1, mod_type="qpsk", t_lateral=0,
                       n_lateral=0):
    n_bits = get_n_bits(mod_type)

    dt = t_symb / n_symb
    if t_lateral != 0 & n_lateral == 0:
        n_lateral = t_lateral / dt

    n_total = len(signal)
    symbols_number = (n_total - 2 * n_lateral) // n_symb
    n_points = symbols_number * n_carriers
    # n_data = n_points * n_bits

    t = np.array([(i - n_total / 2) * dt for i in range(n_total)])

    points = np.zeros(n_points, dtype=complex)

    for k in range(symbols_number):

        # TODO: only for one carrier. add n_car
        for i_car in range(n_carriers):
            one_symbol = np.array([func(i - (k - (num_symbols - 1) / 2) * t_symb, func_args) for i in t])
            signal_integrate = one_symbol * signal
            # we know that n_total = n_symb * symbols_number + 2 * n_lateral
            k_shift = 5  # integrate over k_shift symbols on the left and right
            low_bord = n_lateral + n_symb * (k - k_shift)
            if low_bord < 0:
                low_bord = 0
            up_bord = n_lateral + n_symb * (k + k_shift)
            if up_bord > n_total:
                up_bord = n_total - 1
            one_point = simps(signal_integrate[low_bord:up_bord], t[low_bord:up_bord]) / t_symb

            points[k * n_carriers + i_car] = one_point

    return points


def get_points_wdm(signal, t_symb, n_symb, func, func_args, n_carriers=1, dw=1.0, mod_type="qpsk", t_lateral=0,
                   n_lateral=0):
    n_bits = get_n_bits(mod_type)

    dt = t_symb / n_symb
    if t_lateral != 0 & n_lateral == 0:
        n_lateral = t_lateral / dt

    n_total = len(signal)
    symbols_number = (n_total - 2 * n_lateral) // n_symb
    n_points = symbols_number * n_carriers
    # n_data = n_points * n_bits

    t = np.array([(i - n_total / 2) * dt for i in range(n_total)])

    points = np.zeros(n_points, dtype=complex)

    # go to Fourier space and take values for input frequences
    # for example we had s(t) = \sum_i X_i f(t) e^{i w_i t) -> take s(w_i)

    # TODO: rewrite whole wdm part
    for k in range(symbols_number):
        one_symbol = signal[n_lateral + n_symb * k: n_lateral + n_symb * (k + 1)]
        # print(one_symbol[len(one_symbol) // 2], one_symbol[len(one_symbol) // 2 -1], one_symbol[len(one_symbol) // 2 + 1])
        # print(one_symbol[len(one_symbol) // 2])

        if n_carriers == 1:
            one_point = one_symbol[len(one_symbol) // 2]
            points[k * n_carriers] = one_point
        else:

            for i_carrier in range(n_carriers):
                n_carrier_wave = i_carrier - (n_carriers - (n_carriers % 2)) / 2

                t_symbol = dt * np.array(range(len(one_symbol)))
                one_symbol_moved = one_symbol * np.exp(1.0j * 2.0 * np.pi * dw * n_carrier_wave * t_symbol)

                # find s(w_i)
                one_symbol_spec = fftshift(fft(one_symbol_moved))
                freq = fftshift(fftfreq(len(one_symbol), dt))

                # cut spectrum with width dw near 0
                spectrum_diminishment = 1.0
                spec_filtered, spec_rest = cut_spectrum(one_symbol_spec, freq, dw * spectrum_diminishment)
                one_carrier_restored = ifft(ifftshift(spec_filtered))

                one_point = np.max(one_carrier_restored) / t_symb
                points[k * n_carriers + i_carrier] = one_point

                # matplotlib.rcParams.update({'font.size': 30})
                # fig, axs = plt.subplots(1, 1, figsize=(15, 15))
                # axs.plot(freq, np.absolute(one_symbol_spec), color='blue')
                # axs.plot(freq, np.absolute(spec_filtered), color='red')
                # # axs.set_xlim(-5, 5)
                # axs.set_xlabel('f')
                # axs.set_ylabel('Ampl')
                # axs.grid(True)
                #
                # fig.show()

        # # TODO: only for one carrier. add n_car
        # for i_car in range(n_carriers):
        #     one_symbol = np.array([func(i - (k - (num_symbols - 1) / 2) * t_symb, func_args) for i in t])
        #     signal_integrate = one_symbol * signal
        #     # we know that n_total = n_symb * symbols_number + 2 * n_lateral
        #     k_shift = 5 # integrate over k_shift symbols on the left and right
        #     low_bord = n_lateral + n_symb * (k - k_shift)
        #     if low_bord < 0:
        #         low_bord = 0
        #     up_bord = n_lateral + n_symb * (k + k_shift)
        #     if up_bord > n_total:
        #         up_bord = n_total - 1
        #     one_point = simps(signal_integrate[low_bord:up_bord], t[low_bord:up_bord]) / t_symb
        #
        #
        #     points[k * n_carriers + i_car] = one_point

    return points


# std::vector<std::complex<double>>
# GetConstellationFromWDM(const std::vector<std::complex<double>> signal, int n_carriers, int n_symbol_samples,
#                         double symbol_period, double dw,
#                         std::function<std::complex<double>(double, double)> f_carrier) {
#
#
#     unsigned int n_signal = signal.size();
#     unsigned int n_symbols = n_signal / n_symbol_samples;
#     unsigned int n_data = n_symbols * n_carriers;
#
#     double dt = symbol_period / n_symbol_samples;
#
#     std::vector<std::complex<double>> data_modulated(n_data);
#
#     // Go to Fourier space and take values for input frequences
#     // For example we had s(t) = \sum_i X_i f(t) e^{i w_i t) -> take s(w_i)
#
#
#     for (int i_symbols = 0; i_symbols < n_symbols; ++i_symbols) {
#         std::vector<std::complex<double>> const one_symbol(signal.begin() + i_symbols * n_symbol_samples,
#                                                            signal.begin() + (i_symbols + 1) * n_symbol_samples);
#
#
#         for (int i_carrier = 0; i_carrier < n_carriers; ++i_carrier) {
#
#             int n_carrier_wave = i_carrier - (n_carriers - (n_carriers % 2)) / 2;
#
#             std::vector<std::complex<double>> one_symbol_moved(one_symbol);
#             for (int i = 0; i < n_symbol_samples; i++) {
#                 double t = i * dt;
#                 one_symbol_moved[i] *= std::exp(
#                         std::complex<double>(1i) * 2. * M_PI * dw * (double) n_carrier_wave * t);
#             }
#
#
#             // Convert C++ vector to array for using in GSL
#             gsl_vector_complex *symbol_process = gsl_vector_complex_alloc(n_symbol_samples);
#             symbol_process->data = ComplexVectorToArray(one_symbol_moved);
#             symbol_process->size = n_symbol_samples;
#             symbol_process->stride = 1;
#
#             // Find s(w_i)
#
#             gsl_fft_complex_radix2_forward(symbol_process->data, 1, n_symbol_samples);
#             gsl_vector_complex_scale(symbol_process, gsl_complex_rect(1. / sqrt(n_symbol_samples), 0));
#
# //            DrawAbsSignal(ArrayToComplexVector(symbol_process->data, symbol_process->size));
#
#             // Cut spectrum with width dw near 0
#             double spectrum_diminishment = 1.0;
#             for (int i_spectrum = 1; i_spectrum < n_symbol_samples / 2; ++i_spectrum) {
#                 if (i_spectrum / symbol_period > (dw / 2. * spectrum_diminishment)) {
#                     gsl_vector_complex_set(symbol_process, i_spectrum, gsl_complex_rect(0., 0.));
#                     gsl_vector_complex_set(symbol_process, n_symbol_samples - i_spectrum, gsl_complex_rect(0., 0.));
#                 }
#             }
#
# //            DrawAbsSignal(ArrayToComplexVector(symbol_process->data, symbol_process->size));
#
#             gsl_fft_complex_radix2_backward(symbol_process->data, 1, n_symbol_samples);
#             gsl_vector_complex_scale(symbol_process, gsl_complex_rect(1. / sqrt(n_symbol_samples), 0));
#
#             std::vector<std::complex<double>> one_carrier_restored = ArrayToComplexVector(symbol_process->data,
#                                                                                           symbol_process->size);
#
# //            DrawAbsSignal(one_carrier_restored);
#
#             std::complex<double> data = GetMaxAmplitudePoint(&one_carrier_restored);
# //            std::complex<double> data = one_carrier_restored[one_carrier_restored.size() / 2];
#
#
# //            for (int i = 0; i < n_symbol_samples; i++) {
# //                double t = i * dt;
# //                if (f_carrier(t, symbol_period) != std::complex<double>(0, 0)) {
# //                    data += signal[i + i_symbols * n_symbol_samples] / f_carrier(t, symbol_period) *
# //                            std::exp(-std::complex<double>(1i) * 2. * M_PI * dw * (double) -n_carrier_wave * t) * dt;
# //                }
# //            }
# //            data /= symbol_period;
#
#             data_modulated[i_symbols * n_carriers + i_carrier] = data;
#         }
#     }
#
#     return data_modulated;
# }


def gen_bit_sequence(n_bits, seed=0):
    random.seed(seed)
    bits = random.getrandbits(n_bits)
    data = "{0:b}".format(int(bits))
    if len(data) < n_bits:
        data = ''.join('0' for add_bit in range(n_bits - len(data))) + data

    return data


def gen_wdm_bit_sequence(num_symbols, mod_type, n_carriers=1, seed=0):
    n_bits = n_carriers * get_n_bits(mod_type) * num_symbols
    return gen_bit_sequence(n_bits, seed)


def get_scale_coef_constellation(mod_type):
    return np.max(np.absolute(get_constellation(mod_type)))


def get_scale_coef(points, mod_type):
    return get_scale_coef_constellation(mod_type) / np.max(np.absolute(points))


def get_constellation(mod_type):
    n_points = 2 ** get_n_bits(mod_type)
    points = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        data = "{0:b}".format(int(i))
        points[i] = get_constellation_point(data, mod_type)

    return points


def get_nearest_constellation_point(point, mod_type):
    constellation = get_constellation(mod_type)
    diff = np.absolute(constellation - point * np.ones(len(constellation)))
    ind = np.where(diff == np.amin(diff))
    if len(constellation[ind]) > 1:
        return constellation[ind][0]
    else:
        return constellation[ind]


def get_nearest_constellation_points(points, mod_type):
    points_found = np.zeros(len(points), dtype=complex)
    for i_p in range(len(points)):
        points_found[i_p] = get_nearest_constellation_point(points[i_p], mod_type)
    return points_found


def get_points_error_rate(points_init, points):
    if len(points_init) != len(points):
        print('Error: different bits sequence sizes:', len(points_init), len(points))
        return 1.0

    n = len(points)

    error_count = np.count_nonzero(points - points_init)
    return error_count / n, error_count


def get_bits_error_rate(bits_init, bits):
    if len(bits_init) != len(bits):
        print('Error: different bits sequence sizes:', len(bits_init), len(bits))
        return 1.0

    n = len(bits)

    error_count = 0
    for i in range(n):
        if bits_init[i] != bits[i]:
            error_count += 1

    return error_count / n, error_count


def get_ber_by_points(points_init, points, mod_type):
    bits_init = get_bits_from_constellation_points(points_init, mod_type)
    bits = get_bits_from_constellation_points(points, mod_type)
    return get_bits_error_rate(bits_init, bits)


# nonlinear phase compensation (phase shift equalisation)
def make_pse(points_init, points, mod_type):

    n = 100
    d_alpha = 2 * np.pi / n
    min_per = 1.0
    i_pos = 0

    phi = np.angle(points)
    r = np.absolute(points)

    for i in range(n):
        point_temp = r * (np.cos(phi + d_alpha * i) + 1j * np.sin(phi + d_alpha * i))
        point_found_r = np.zeros(len(point_temp), dtype=complex)
        for i_p in range(len(point_temp)):
            point_found_r[i_p] = get_nearest_constellation_point(point_temp[i_p], mod_type)
        per = get_points_error_rate(points_init, point_found_r)[0]
        if per < min_per:
            min_per = per
            i_pos = i

    points_rotated = r * (np.cos(phi + d_alpha * i_pos) + 1j * np.sin(phi + d_alpha * i_pos))
    return points_rotated


def convert_power_dl_mw(beta2, t_symb):
    ...


def convert_power_mw_dl():
    ...


def convert_power_mw_dbm():
    ...


def convert_power_dbm_mw():
    ...


def dispersion_to_beta2(dispersion, wavelenght_nm=1550):
    # dispersion in ps/nm/km, wavelenght_nm in nm
    return -(wavelenght_nm ** 2) * (dispersion * 10 ** 3) / (2. * np.pi * 3.0 * 10 ** 8)


def nd_to_mw(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    # t_symb in ps, beta2 in ps^2/km, gamma in mW^-1 * km^-1
    return p * beta2 / gamma * (t_symb) ** (-2)


def mw_to_nd(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    # t_symb in ps, beta2 in ps^2/km, gamma in mW^-1 * km^-1
    return p / (beta2 / gamma * (t_symb) ** (-2))


def mw_to_dbm(p):
    return 10 * np.log10(p)


def dbm_to_mw(p):
    return 10 ** (p / 10)


def nd_to_dbm(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    return mw_to_dbm(nd_to_mw(p, t_symb, beta2, gamma))


def dbm_to_nd(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    return mw_to_nd(dbm_to_mw(p), t_symb, beta2, gamma)


def z_nd_to_km(z_nd, t_symb=100, beta2=21.5):
    return z_nd * t_symb ** 2 / beta2


def z_km_to_nd(z_km, t_symb=100, beta2=21.5):
    return z_km / (t_symb ** 2 / beta2)


def add_lateral(signal, n=0, value=0.0):
    np_signal = len(signal)
    if n == 0:
        n = np_signal // 2

    new_signal = value * np.ones(np_signal + 2 * n, dtype=complex)
    new_signal[n : n + np_signal] = signal

    return new_signal


def add_lateral_function(signal, fun, dt, n=0):
    np_signal = len(signal)
    if n == 0:
        n = np_signal // 2


    left_point = signal[0]
    fun_param_left = [dt, abs(abs(signal[1]) - abs(signal[0])), left_point, -dt]
    right_point = signal[-1]
    fun_param_right = [dt, -abs(abs(signal[-1]) - abs(signal[-2])), right_point, dt]
    # left = np.array([left_point * fun(i * dt, fun_param_left) for i in range(-n + 1, 1, 1)])
    left = np.array([fun(i * dt, fun_param_left) for i in range(-n + 1, 1, 1)])
    # right = np.array([right_point * fun(i * dt, fun_param_right) for i in range(n)])
    right = np.array([fun(i * dt, fun_param_right) for i in range(n)])
    new_signal = np.concatenate((left, signal, right), axis=None)

    return new_signal


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def round_power_of_2(order, x):
    temp = next_power_of_2(x)
    for k in range(order - 1):
        temp = next_power_of_2(temp + 1)

    return temp



if __name__ == "__main__":
    mod_type = "16qam"
    n_car = 1
    t_symb = 1.0
    n_symb = 16
    dt = t_symb / n_symb

    num_symbols = 128
    n_lateral = 0 * n_symb
    # p_ave = 0.74
    p_ave = 1
    roll_off = 0.01

    data = gen_wdm_bit_sequence(num_symbols, mod_type, n_car)
    points = get_constellation_point(data, mod_type)
    signal = get_wdm_signal(data, t_symb=t_symb, n_symb=n_symb,
                            func=rcos, func_args=[t_symb, roll_off],
                            n_carriers=n_car, mod_type=mod_type, n_lateral=n_lateral)

    points_restore = get_points_wdm(signal, t_symb=t_symb, n_symb=n_symb,
                                    func=rcos, func_args=[t_symb, roll_off],
                                    n_carriers=n_car, mod_type=mod_type, n_lateral=n_lateral)

    scale = get_scale_coef(points_restore, mod_type)
    points_restore = scale * points_restore

    matplotlib.rcParams.update({'font.size': 30})

    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(np.real(points), np.imag(points),
                marker='o', color='blue', ls='')
    axs[0].plot(np.real(points_restore), np.imag(points_restore),
                marker='*', color='red', ls='')
    # axs[0].set_xlim(-5, 5)
    axs[0].set_xlabel('Re')
    axs[0].set_ylabel('Im')
    axs[0].grid(True)

    axs[1].plot(np.real(points - points_restore), np.imag(points - points_restore),
                marker='o', color='blue', ls='')
    axs[1].set_xlabel('Re')
    axs[1].set_ylabel('Im')
    axs[1].grid(True)

    fig.show()

    # point_c = get_constellation(mod_type)
    # fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    # axs.plot(np.real(point_c), np.imag(point_c),
    #             marker='o', color='blue', ls='')
    # # axs[0].set_xlim(-5, 5)
    # axs.set_xlabel('Re')
    # axs.set_ylabel('Im')
    # axs.grid(True)
    #
    # fig.show()

# Example
# import random
# import time
#
# start = time.time()
#
# t_symb = 1.
# n_symb = 2**5
# dt = t_symb / n_symb
#
# p = 0.1
# num_symbols = 10
# mod_type = "qpsk"
# n_car = 1
# lat = 4*n_symb
#
# n_bits = n_car * get_n_bits(mod_type) * num_symbols
# bits = random.getrandbits(n_bits)
# data = "{0:b}".format(int(bits))
# if len(data) < n_bits:
#     data = ''.join('0' for add_bit in range(n_bits - len(data))) + data
#
# # print(len(data), data)
#
# signal = get_wdm_signal(data, t_symb, n_symb, rcos, [t_symb, p], n_carriers=n_car, mod_type=mod_type, n_lateral=lat)
# print(get_average_power(signal, dt))
# signal = set_average_power(signal, dt, 1.0)
# print(get_average_power(signal, dt))
#
# t = np.array([i * dt for i in range(len(signal))])
# print(signal[0], signal[-1])
#
# print("Number of points in signal:", len(signal))
# end = time.time()
# print("Elapsed = %s" % (end - start))
#
#
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(t, np.power(np.absolute(signal), 2), 'blue')
# # axs[0].plot(t, np.power(np.absolute(one_symbol), 2), 'red')
# # axs[0].plot(t, np.power(np.absolute(rcos_signal), 2), 'red')
# axs[0].set_xlim(0, dt * (n_symb * num_symbols + 2*lat))
# # axs[0].set_xlim(-5, 5)
# axs[0].set_xlabel('Time')
# axs[0].set_ylabel('Power')
# axs[0].grid(True)
# #
# axs[1].plot(np.power(np.abs(fftshift(fft(signal))), 2), 'blue')
# # axs[1].plot(np.power(np.abs(fftshift(fft(one_symbol))), 2), 'red')
# # # axs[1].plot(freq, np.power(np.abs(fftshift(fft(rcos_signal))), 2), 'red')
# # # axs[1].plot(freq, np.power(np.abs(rcos_spectrum * nt / t_span), 2), 'green')
# # # # axs[1].set_xlim(-np.pi / t_span * nt / (2 * np.pi), np.pi / t_span * nt / (2 * np.pi))
# axs[1].set_xlim(len(signal) * 0.45, len(signal) * 0.55)
# # axs[1].set_xlabel('Normalized Frequency')
# # axs[1].set_ylabel('Spectral Power')
# axs[1].grid(True)
#
# fig.show()
# print("done")

# nt = 2 ** 12
# t_span = 2 ** 6
# dt = t_span / nt
# t = np.array([(i - nt / 2) * dt for i in range(nt)])
# w = np.array([(i - nt / 2) * (2. * np.pi / t_span) for i in range(nt)])
# freq = w / (2 * np.pi)  # freq. array
#
# t_symb = 1.
# sinc_signal = np.sinc(t / t_symb)
# p = 0.1
# rcos_signal = np.array([rcos(i, t_symb, p) for i in t])
# rcos_spectrum = np.array([rcos_spec(i, t_symb, p) for i in freq])
# # print(rcos_spectrum, np.max(rcos_spectrum))
#
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(t, np.power(np.absolute(sinc_signal), 2), 'blue')
# axs[0].plot(t, np.power(np.absolute(rcos_signal), 2), 'red')
# # axs[0].set_xlim(-t_span / 2, t_span / 2)
# axs[0].set_xlim(-5, 5)
# axs[0].set_xlabel('Time')
# axs[0].set_ylabel('Power')
# axs[0].grid(True)
#
# axs[1].plot(freq, np.power(np.abs(fftshift(fft(sinc_signal))), 2), 'blue')
# axs[1].plot(freq, np.power(np.abs(fftshift(fft(rcos_signal))), 2), 'red')
# axs[1].plot(freq, np.power(np.abs(rcos_spectrum * nt / t_span), 2), 'green')
# # axs[1].set_xlim(-np.pi / t_span * nt / (2 * np.pi), np.pi / t_span * nt / (2 * np.pi))
# axs[1].set_xlim(-2, 2)
# axs[1].set_xlabel('Normalized Frequency')
# axs[1].set_ylabel('Spectral Power')
# axs[1].grid(True)
#
# fig.show()


# nt = 2 ** 12
# t_span = 2 ** 6
# dt = t_span / nt
# t = np.array([(i - nt / 2) * dt for i in range(nt)])
# w = np.array([(i - nt / 2) * (2. * np.pi / t_span) for i in range(nt)])
# freq = w / (2 * np.pi)  # freq. array
#
# t_symb = 1.
# sinc_signal = np.sinc(t / t_symb)
# p = 0.1
# rcos_signal = np.array([rcos(i, t_symb, p) for i in t])
# rcos_spectrum = np.array([rcos_spec(i, t_symb, p) for i in freq])
# # print(rcos_spectrum, np.max(rcos_spectrum))
#
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(t, np.power(np.absolute(sinc_signal), 2), 'blue')
# axs[0].plot(t, np.power(np.absolute(rcos_signal), 2), 'red')
# # axs[0].set_xlim(-t_span / 2, t_span / 2)
# axs[0].set_xlim(-5, 5)
# axs[0].set_xlabel('Time')
# axs[0].set_ylabel('Power')
# axs[0].grid(True)
#
# axs[1].plot(freq, np.power(np.abs(fftshift(fft(sinc_signal))), 2), 'blue')
# axs[1].plot(freq, np.power(np.abs(fftshift(fft(rcos_signal))), 2), 'red')
# axs[1].plot(freq, np.power(np.abs(rcos_spectrum * nt / t_span), 2), 'green')
# # axs[1].set_xlim(-np.pi / t_span * nt / (2 * np.pi), np.pi / t_span * nt / (2 * np.pi))
# axs[1].set_xlim(-2, 2)
# axs[1].set_xlabel('Normalized Frequency')
# axs[1].set_ylabel('Spectral Power')
# axs[1].grid(True)
#
# fig.show()

# draw constellation
# points_b = []
# points = []
# for i in range(64):
#     a = "{0:b}".format(int(i))
#     # print(a)
#     # print(get_constellation_point(a, "64qam"))
#     points_b.append(a)
#     points.append(get_constellation_point(a, "64qam"))
#
# print(points_b)
# print(points)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# p = ax.plot(np.real(points), np.imag(points), 'o')
# ax.set_xlabel('real part')
# ax.set_ylabel('imag part')
# ax.set_title('Constellation')
# fig.show()

# get_wdm_symbol(0, 0, rcos, [1., 0.9], 0)

# n_bits = {"qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8, "1024qam": 10}
# print(n_bits["16qam"])
