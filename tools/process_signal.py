import sys
sys.path.insert(0, '/home/esf0/PycharmProjects/nft_processing')

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime
from importlib import reload

from prettytable import PrettyTable
from scipy.fft import fftshift, ifftshift, fft, ifft

import hpcom
from hpcom.signal import create_wdm_parameters, generate_wdm, get_points_wdm, receiver_wdm, nonlinear_shift, rrcosfilter, \
    update_wdm_parameters_from_json
from hpcom.channel import create_channel_parameters, update_channel_parameters_from_json
from hpcom.metrics import get_evm

# from ssfm_gpu.propagation import propagate_manakov, dispersion_compensation_manakov
from ssfm_gpu.propagation import propagate_schrodinger, dispersion_compensation
from ssfm_gpu.conversion import convert_forward, convert_inverse

# import signal_handling.processing as prcs
from signal_handling.processing import get_default_process_parameters, example_nlse_processing, nlse_tx_signal_data
import functions as fn

import FNFTpy as fpy
import PJTpy as pjt

# reload(prcs)
# reload(hpcom)



from hpcom.signal import create_wdm_parameters, generate_wdm, get_points_wdm, receiver_wdm, nonlinear_shift, rrcosfilter


wdm_json = 'wdm_parameters.json'
channel_json = 'channel_parameters.json'

data_dir = 'data/'
job_name = 'process_signal_test'

mode = 'tx'
save_flag = True
n_iter_save = 10
omp_num_threads = 12

wdm = update_wdm_parameters_from_json(wdm_json)
channel = update_channel_parameters_from_json(channel_json)

process_parameters = get_default_process_parameters()
process_parameters['window_mode'] = 'cdc'
# process_parameters['n_steps'] = 256
process_parameters['n_steps'] = 3
process_parameters['n_symb_proc'] = 32
process_parameters['n_symb_side'] = 300
process_parameters['n_symb_total'] = process_parameters['n_symb_proc'] + 2 * process_parameters['n_symb_side']
# process_parameters['n_symb_skip'] = 1024
process_parameters['n_symb_skip'] = 1224
process_parameters['n_symb_add'] = fn.side_to_np2(process_parameters['n_symb_proc'] + 2 * process_parameters[
    'n_symb_side'])  # add to total number of symbols to make it power of 2


if __name__ == '__main__':

    print('Starting...')

    # Allocate GPU mempory if there any GPU available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    print('List of GPUs:', gpus)

    if mode == 'rx':
        result = example_nlse_processing(wdm, channel, process_parameters,
                                         omp_num_threads=omp_num_threads,
                                         job_name=job_name,
                                         save_flag=save_flag, dir=data_dir, n_iter_save=n_iter_save)
        points = result['points']
        points_nft = result['points_nft']
        evms = result['evm']
    elif mode == 'tx':
        result = nlse_tx_signal_data(wdm, channel, process_parameters,
                                     omp_num_threads=omp_num_threads,
                                     job_name=job_name,
                                     save_flag=save_flag, dir=data_dir, n_iter_save=n_iter_save)

    print('Finished!')
