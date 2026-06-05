
"""
This script produces the PCA scatter plot showing how PV, SST, and VIP synapses distribute in SRP parameter space.

Author: jadepoir
"""

import copy
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())

fits = pickle.load(open("./Data/processed_input_in_rodent.p", "rb"))

srp_fits = [pair[8:-2] for pair in fits]

pretypes_count = Counter([pair[0] for pair in fits])
count_pv = pretypes_count['pvalb']
count_sst = pretypes_count['sst']
count_vip = pretypes_count['vip']

scaler.fit(srp_fits)
scaled_arr = scaler.transform(srp_fits)
pca_result = pca.fit_transform(scaled_arr)

fig, ax = plt.subplots()

ax.scatter(pca_result[:, 0][:count_pv], pca_result[:, 1][:count_pv], color="xkcd:dark blue", alpha=0.5, s=8, edgecolor='none', label='pvalb')
ax.scatter(pca_result[:, 0][count_pv:count_pv + count_sst], pca_result[:, 1][count_pv:count_pv + count_sst], color="xkcd:red orange", alpha=0.5, s=8, edgecolor='none', label='sst')
ax.scatter(pca_result[:, 0][count_pv + count_sst:], pca_result[:, 1][count_pv + count_sst:], color="xkcd:golden yellow", alpha=0.5, s=8, edgecolor='none', label='vip')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(fontsize=6)
ax.tick_params(labelsize=6)
fig.set_dpi(1200)
# fig.set_size_inches(1.911, 1.352)
plt.savefig("./Figures/fig2/Scatter_pretypes.svg", transparent=True)
# plt.show()