import numpy as np


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def side_to_np2(n):
    return int(next_power_of_2(n) - n) // 2


def form_name(wdm, channel):
    return f'_{wdm["m_order"]}qam_{int(wdm["symb_freq"] / 1e9)}gbd_{channel["n_spans"]}spans_{channel["z_span"]}km'


# Function to format x-ticks as 2 in power something
def format_ticks(x, pos):
    return f'$10^{{{int(np.log10(x))}}}$'


# Define a custom tick formatter function
def custom_sci_notation(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    return r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)