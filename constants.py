import os


def _create_dic(dx, dist, start, nb):

    dic = {}
    dic['dx'] = dx
    dic['dist'] = dist
    dic['start'] = start
    dic['nb'] = nb

    return dic


############ Frequently used DATASET SETUPS.
# Defines surface thresholds, data resolution and (number of) scales
# _create_dic entries as follows:
#
# (1) resolution, (2) distance between scales, (3) start scale, (4) number of scales

NAMES = {
    'SM5k': _create_dic(5, 0.25, 5, 35),  # to avoid bias in full wavelet reconstruction, (2) = 0.25 and a minimum of 35 scales is needed.
    'SM5k_sensiTest': _create_dic(5, 0.25, 5, 20),
    'SM_testUM': _create_dic(1, 0.25, 2, 40),
    'SM_testUM2': _create_dic(1, 0.5, 2, 40),
    'SM_sensiTestUM': _create_dic(1, 0.25, 2, 20),
    'SM1p5km_control': _create_dic(1.5, 0.25, 4, 40),
    'SM4km_control': _create_dic(4, 0.25, 5, 40),
    'SM1p5km_corecomp': _create_dic(1.5, 0.25, 4, 30),
}

########## Test case data

TESTDATA = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'testdata' + os.sep + 'sm_testfile.nc'
