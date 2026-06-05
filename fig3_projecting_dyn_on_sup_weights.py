
"""
Computes supervised classifier–derived feature‑weight axes and projects SRP model parameters onto them.

Author: jadepoir
"""

import os
import copy
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pickle.load(open('./Data/supervised_alg_in_rodent_model.p', 'rb'))

params = {
    "mu_baseline": [],
    "mu_A1": [],
    "mu_A2": [],
    "mu_A3": [],
    "mu_A4": [],
    "sigma_baseline": [],
    "sigma_A": [],
}

weights = {
    "pv": params.copy(),
    "sst": params.copy(),
    "vip": params.copy()
}

for i, pre_type in enumerate(weights.keys()):
    for j, param in enumerate(params.keys()):
        weights[pre_type][param] = np.nanmean(data[2][i][j])

w_matrix = np.array([[weights["pv"][param], weights["sst"][param], weights["vip"][param]] for param in params.keys()])

pv_v = w_matrix[:, 0].copy()
sst_v = w_matrix[:, 1].copy()
vip_v = w_matrix[:, 2].copy()

v1 = pv_v.copy()
v2 = vip_v - (np.dot(v1, vip_v) / np.dot(v1, v1)) * v1
v3 = sst_v - (np.dot(v2, sst_v) / np.dot(v2, v2)) * v2 - (np.dot(v1, sst_v) / np.dot(v1, v1)) * v1

pc_weights = np.array([v1, v2, v3]).T

fits = os.listdir("./Data/srp_fits_in_rodent")

srp_fits_pre = [pickle.load(open("./Data/srp_fits_in_rodent/" + pair, "rb")) for pair in fits]

index_to_delete = []

srp_fits = []
for i, fit in enumerate(srp_fits_pre):
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus = fit
    if sigma_baseline == 6 and sigma_amps == 1000:
        index_to_delete.append(i)
        continue
    srp_fits.append([mu_baseline, *mu_amps, sigma_baseline, sigma_amps])

srp_fits = np.array(srp_fits)

pretypes_count = Counter([pair.split("_")[0] for pair in fits if fits.index(pair) not in index_to_delete])
print(pretypes_count)
count_pv = pretypes_count['pvalb']
count_sst = pretypes_count['sst']
count_vip = pretypes_count['vip']

scaler = copy.deepcopy(StandardScaler())
scaler.fit(srp_fits)
scaled_arr = scaler.transform(srp_fits)
pca_result = srp_fits @ pc_weights

print(pc_weights)

fig, ax = plt.subplots()

ax.scatter(pca_result[:, 0][:count_pv], pca_result[:, 1][:count_pv], color="xkcd:dark blue", alpha=0.5, marker=".", s=8, edgecolor='none')
ax.scatter(pca_result[:, 0][count_pv:count_pv + count_sst], pca_result[:, 1][count_pv:count_pv + count_sst], color="xkcd:red orange", alpha=0.5, marker=".", s=8, edgecolor='none')
ax.scatter(pca_result[:, 0][count_pv + count_sst:], pca_result[:, 1][count_pv + count_sst:], color="xkcd:golden yellow", alpha=0.5, marker=".", s=8, edgecolor='none')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=6)
ax.set_ylim(-100, 100)
ax.set_xlim(-500, 500)
fig.set_dpi(1200)
fig.set_size_inches(2.719, 1.602)
plt.savefig("./Figures/fig3/Scatter_pretypes_supervised_weights.svg", transparent=True)
# plt.show()