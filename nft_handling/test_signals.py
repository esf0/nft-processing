import numpy as np
from scipy.special import gamma
from scipy.linalg import expm, inv

############################################
# Test signals for NLSE (one polarisation) #
############################################


def get_sech(t, xi, a, c, sigma=1):
    q = a * np.power(1.0 / np.cosh(t), (1.0 + 1.0j * c))
    d = np.sqrt(sigma * a ** 2 - c ** 2 / 4.0)

    a_xi = gamma(0.5 - 1.0j * (xi + c / 2)) * gamma(0.5 - 1.0j * (xi - c / 2)) / \
           (gamma(0.5 - 1.0j * xi - d) * gamma(0.5 - 1.0j * xi + d))

    b_xi = 1.0 / (2.0 ** (1.0j * c) * a) * gamma(0.5 - 1.0j * (xi + c / 2)) * gamma(0.5 + 1.0j * (xi - c / 2)) / \
           (gamma(-0.5j * c - d) * gamma(-0.5j * c + d))

    k = np.array([i for i in range(int(np.sqrt(a ** 2 - c ** 2 / 4.0) - 0.5) + 1)])

    xi_discr = 1.0j * (np.sqrt(a ** 2 - c ** 2 / 4.0) - 0.5 - k)

    f = gamma(0.5 - 1.0j * (xi_discr + c / 2.0)) * gamma(0.5 - 1.0j * (xi_discr - c / 2.0)) / gamma(
        0.5 - 1.0j * xi_discr + d)

    phi = np.ones(len(k), dtype=complex)
    # for i in range(k[-1] - 1):
    for l in range(len(k) - 1):
        phi[l + 1] = -(l + 1) * phi[l]
        # print('phi', l+1, phi[l+1])

    b_discr = 1.0 / (2 ** (1.0j * c) * a) * gamma(0.5 - 1.0j * (xi_discr + c / 2)) * gamma(
        0.5 + 1.0j * (xi_discr - c / 2)) / \
              (gamma(-0.5j * c - d) * gamma(-0.5j * c + d))
    r_discr = 1.0j * b_discr / f / phi
    ad_discr = f * phi / 1.0j

    return q, a_xi, b_xi, xi_discr, b_discr, r_discr, ad_discr


def get_sech_shape(t, a, c, sigma=1):
    q, _, _, _, _, _, _ = get_sech(t, np.array([0]), a, c, sigma)
    return q


def get_sech_b_coef(xi, a, c, sigma=1):
    d = np.sqrt(sigma * a ** 2 - c ** 2 / 4.0)
    b_xi = 1.0 / (2.0 ** (1.0j * c) * a) * gamma(0.5 - 1.0j * (xi + c / 2)) * gamma(0.5 + 1.0j * (xi - c / 2)) / \
           (gamma(-0.5j * c - d) * gamma(-0.5j * c + d))

    return b_xi


#########################################################
# Test signals for Manakov equation (two polarisations) #
#########################################################


def get_rect_dual(t, t1, t2, xi, a1, a2, sigma=1):

    transfer_rect = np.array([[-1j * xi, a1, a2],
                              [-sigma * np.conj(a1), 1j * xi, 0.],
                              [-sigma * np.conj(a2), 0., 1j * xi]])
    transfer_free = np.array([[-1j * xi, 0., 0.],
                              [0., 1j * xi, 0.],
                              [0., 0., 1j * xi]])
    start_jost = np.array()

    np.matmul(np.matmul(inv(expm(transfer_free * t2)), expm(transfer_rect * t2)),
              np.matmul(inv(expm(transfer_rect * t1)), start_jost))

    return 0


def get_sech_dual(t, xi, a1, a2, sigma=1):

    q1 = a1 * 1.0 / np.cosh(t)
    q2 = a2 * 1.0 / np.cosh(t)

    d = np.sqrt(sigma * (a1 ** 2 + a2 ** 2))

    a_xi = gamma(0.5 - 1.0j * xi) * gamma(0.5 - 1.0j * xi) / \
           (gamma(0.5 - 1.0j * xi - d) * gamma(0.5 - 1.0j * xi + d))

    b1_xi = -sigma * a1 * np.sin(np.pi * d) / (d * np.cosh(np.pi * xi))
    b2_xi = -sigma * a2 * np.sin(np.pi * d) / (d * np.cosh(np.pi * xi))





    return 0
