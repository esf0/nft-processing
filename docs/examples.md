# Examples

## Signal generation

In this module, we write a bit non-standard realisation of signal generation.
I propose to use your own signal generator, cause my is not efficient and have a lot of redundant functionality.  
But if you still wand to use it, here is an example:  
```
modulation_type = "16qam"  # modulation
n_car = 1  # number of subcarriers (only 1 now)
t_symb = 1.  # dimensionless time interval

number_of_symbols = 2 ** 14  # total number of symbols to generate
np_symb = 16  # samples per symbol
n_lateral_symbols = 32  # number of time slots (symbols) on the left and right for the signal
# I added it because pulse carrier shape can have very long tails in time domain
n_lateral = n_lateral_symbols * np_symb  # number of points for lateral intervals (left and right)

z_80km_nd = sg.z_km_to_nd(80, t_symb=14.8)  # 80km in dimensionless units
p_dbm = -4  # signal average power in dBm
dt = t_symb / np_symb  # time step

# generate signal with defined parameters
q = sg.create_signal(number_of_symbols, p_dbm, modulation_type, n_car, t_symb=t_symb, np_symb=np_symb, n_lateral_symbols=n_lateral_symbols)
n_add = 2 ** 10 * np_symb  #  it is additional zeros on the left and right side of the signal. 2^10 symbols * number of point per symbol
# n_add = 2 ** 6 * np_symb
q = sg.add_lateral(q, n_add)

# create time
n_t = len(q)
# t = np.array([(i - (n_t - 1) / 2) * dt for i in range(n_t)])
t = np.array([(i - n_t / 2) * dt for i in range(n_t)])
t_span = t[-1] - t[0]
```  

That's it, you have created signal with side intervals. So now you have your signal with `n_add + n_lateral` 
points on the right and left.  
Such non-standard signal generation is historical heritage of some numerical experiments.  
So, I highly recommend use your own generator.  

This signal will have side intervals. You can loop signal with
```
# sg is signal_generation module
q = sg.loop_signal(q, n_add + n_lateral)
```
This function cuts first and last `n_add + n_lateral` and adds first points to the end. 
And last cutted points adds to the beginning.  
In such case, you can set n_add in signal generation section `n_add=0` cause this points are zeros.  
If it is unclear, here is the definition:
```
def loop_signal(q, n_add):

    q_central = np.copy(q[n_add: -n_add])
    n = len(q_central)
    for k in range(n_add):
        q_central[k] += q[n + n_add + k]
        q_central[-k - 1] += q[n_add - 1 - k]

    return q_central
```

Also, we use constellation scaling:  
```
points_restored = np.array(sg.get_points_wdm(q, t_symb, np_symb, None, None, n_carriers=n_car, mod_type=modulation_type, n_lateral=n_lateral + n_add))
constellation_scale = sg.get_scale_coef(points_restored, modulation_type)
points_restored = constellation_scale * points_restored
```  
`constellation_scale` coefficient is used to scale constellation and calculate BER.


## Signal processing

We created or generated signal. We now can calculate propagation of the signal.
```
load_flag = False

z_80km_nd = sg.z_km_to_nd(80, t_symb=14.8)
file_name = '../data/propagated_signals/wdm_prop_example.npy'

# if load_flag == False then calculate propagation
if not load_flag:
    n_z_span = 12
    # ssfm propagation
    beta2 = -1.0
    gamma = 1.0

    q_prop_total = np.zeros((n_z_span + 1, len(q)), dtype=complex)  # contains all signals for n_z_span distances
    q_prop_total[0] = q
    q_prop = q


    n_z_prop = 2 ** 9
    for k_span in range(n_z_span):
        # time_start = time.time()
        q_prop = ssfm.fiber_propogate(q_prop, t_span, z_80km_nd, n_z_prop, gamma=gamma, beta2=beta2)
        q_prop_total[k_span + 1] = q_prop
        # time_end = time.time()
        # print('Time, s:', time_end - time_start)


    with open(file_name, 'wb') as f:
        np.save(f, q_prop_total)

else:
    with open(file_name, 'rb') as f:
        q_prop_total = np.load(f)
    q = q_prop_total[0]
```
If you set `load_flag = False` than you will use SSFM to calculate propagation for `n_z_span` spans.
Each span if equal to 80km of real fiber in dimensionless units `z_80km_nd`.  
For every span we store propagated signal in `q_prop_total[k_span]` where `k_span` is number of propagated spans.
For example, `q_prop_total[0]` contains initial signal, `q_prop_total[1]` -- signal after one span, etc.  
You can change accuracy using `n_z_prop` -- number of point for span.  
Results will be saved to `file_name` file.

If you load signal with `load_flag = True`, you have to specify again some variables (if you don't loop the signal):  
```
n_add = 2 ** 10 * np_symb  # or what you used before. If you loop signal, it is zero.

n_t = len(q)
t = np.array([(i - n_t / 2) * dt for i in range(n_t)])
t_span = t[-1] - t[0]
```  

Now we can process signal with 
```
n_z_span_proc = 0
q_proc = q_prop_total[n_z_span_proc]
z_prop = n_z_span_proc * z_80km_nd

result = sp.process_signal(q, q_proc, z_prop, np_symb, t_symb, n_car, modulation_type, constellation_scale, n_add=n_add + n_lateral, print_sys_message=True)
```

This procedure takes signal and makes DBP with NFT. 
Restore signal, restore points from WDM signal.
Also, it compares NFT-DBP with CDC. But it doesn't make phase rotation.

## NFT routines

There are many NFT routines that you can use.  
See documentation for [nft_analyse.py](nft_analyse.html)  

The high-level interface is represented in the following functions:
* `nft_analyse.get_discrete_eigenvalues(...)`
* `nft.get_continuous_spectrum(...)`
* `nft.get_discrete_spectrum(...)`
* `nft.get_discrete_spectrum_coefficients(...)`

To perform DBP you can use 
```
res = nft_analyse.make_dbp_nft(q, t, z_back)
```
with additional parameters, which define NFT procedures.  

Types of schemes for FNFT you can find in [auxilary.py](auxilary.html).  

Example of use:  
```
ampl = 2
chirp = 0.1

t_span = 64.0
n_t = 2 ** 10

dt = t_span / (n_t - 1)
t = np.array([i * dt - t_span / 2. for i in range(n_t)])

xi_span = np.pi / dt
n_xi = 2 ** 10
d_xi = xi_span / (n_xi - 1)
xi = np.array([i * d_xi - xi_span / 2. for i in range(n_xi)])

q, a_xi, b_xi, xi_discr, b_discr, r_discr, ad_discr = test_signals.get_sech(t, xi, a=ampl, c=chirp)
print(len(xi_discr), xi_discr)
z_prop = 4.3
q_prop = ssfm.fiber_propogate(q, t_span, z_prop, n_span=2 ** 9, gamma=1, beta2=-1)

z_back = z_prop
xi_upsampling = 1
forward_continuous_type='fnft'
forward_discrete_type='fnft'
forward_discrete_coef_type='bi-direct'
inverse_type='both'
fnft_type=0
nft_type='bo'
use_contour = False
n_discrete_skip = 2

res_dbp = nft.make_dbp_nft(q_prop, t, z_back, xi_upsampling=xi_upsampling,
                           forward_continuous_type=forward_continuous_type,
                           forward_discrete_type=forward_discrete_type,
                           forward_discrete_coef_type=forward_discrete_coef_type,
                           inverse_type=inverse_type,
                           fnft_type=fnft_type, nft_type=nft_type,
                           use_contour=use_contour, n_discrete_skip=n_discrete_skip,
                           print_sys_message=True)

q_tib_total = res_dbp['q_total']
q_tib_left = res_dbp['q_tib_left']
q_tib_right = res_dbp['q_tib_right']
q_fnft = res_dbp['q_fnft']
```