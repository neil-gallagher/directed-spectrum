import sys
from pathlib import Path
import numpy as np

home_dir = str(Path.home())
pipepath = home_dir + '/lpne-data-analysis'
sys.path.append(pipepath)
from data_tools import load_data
from dir_spec import ds

# load timeseries data
X, labels = load_data(pipepath+'/TeST.mat', feature_list=('X'))
X = np.asarray(X)

# shape X and extract 1 window
X_win = X[0]

# calculate DS
win_len = np.rint(labels['fs'] * 2/5)
is_nac = [r[:3] == 'Acb' for r in labels['area']]
is_snc = [r[1:] == 'SNC' for r in labels['area']]
is_dhip = [r[1:] == 'DHip' for r in labels['area']]
groups = np.where(is_nac, 'Nac', labels['area'])
groups = np.where(is_snc, 'SNC', groups)
groups = np.where(is_dhip, 'DHip', groups)
dir_spec = ds.ds(X_win, labels['fs'], groups, fres=1, window='boxcar',
                 nperseg=win_len)
