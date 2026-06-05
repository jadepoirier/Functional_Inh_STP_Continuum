
"""
Generates all analyses and visualizations for Figure 4 by constructing a probabilistic
continuum of cell‑type identity, plotting medoid kernels and plotting parameter distributions.

Author: jadepoir
"""

import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from srplasticity.srp import easySRP
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import Isomap
from matplotlib.ticker import MaxNLocator

data = pickle.load(open('./Data/supervised_alg_in_rodent_model.p', 'rb'))
# data = pickle.load(open("./Data/supervised_alg_in_rodent_phys.p", "rb"))

accuracies = data[0] # Level 1: Seven algs; Level 2: Accuracies, baseline, differences
alg_labels = data[1] # Seven algs
coef_outputs = data[2] # Level 1: 3 pre types; Level 2: 7 params
perfo_outputs = data[3] # Level 1: Seven algs; Level 2: 3 pre types
final_prob_dict = data[4]

# Define base colors using XKCD color names
xkcd_colors = mcolors.XKCD_COLORS
color1 = np.array(mcolors.to_rgb(xkcd_colors['xkcd:dark blue']))
color2 = np.array(mcolors.to_rgb(xkcd_colors['xkcd:red orange']))
color3 = np.array(mcolors.to_rgb(xkcd_colors['xkcd:golden yellow']))

color_dict = {'pvalb': color1,
              'sst': color2,
              'vip': color3}


def plot_kernel_easySRP(axis, model, colour="#03719c"):
    """
    Plot the efficacy kernel on the given axis.

    :param axis: The axis on which to plot the kernel
    :type axis: matplotlib.axes.Axes
    :param model: The SRP model with history dependent mean behaviour and fixed variance
    :type model: class: 'easySRP'
    :param colour: Colour of the kernel plot. Defaults to #03719c
    :type colour: str, optional
    """

    if model.__class__.__name__ != 'easySRP':
        raise ValueError("'model' must be an instance of easySRP")

    kernel_y = model.run_ISIvec([200, 801], fast=False, return_all=True)["filtered_spiketrain"][:10000]
    kernel_x = np.arange(0, 2000, 0.1)[:10000] - 200

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.plot(kernel_x, kernel_y, color=colour)
    axis.set_ylabel("Kernel", labelpad=1)
    axis.set_xlabel("Time (ms)", labelpad=1)

def soft_confusion_per_cluster(cluster_assignments,
                               true_labels,
                               predicted_probs,
                               n_clusters,
                               n_classes):
    """
    cluster_assignments : array of shape (N,)
        Cluster index for each synapse (0..n_clusters-1)
    true_labels : array of shape (N,)
        Integer true labels (0..n_classes-1)
    predicted_probs : array of shape (N, n_classes)
        Model probability vectors for each synapse
    n_clusters : int
        Number of clusters
    n_classes : int
        Number of cell types (e.g., 3)
    """

    # Convert true labels to one-hot
    true_onehot = np.eye(n_classes)[true_labels]

    # Store results
    cluster_soft_confusion = {}

    for c in range(n_clusters):
        # indices of synapses in this cluster
        idx = np.where(cluster_assignments == c)[0]

        if len(idx) == 0:
            cluster_soft_confusion[c] = np.zeros((n_classes, n_classes))
            continue

        # true and predicted for this cluster
        T = true_onehot[idx]          # shape (Nc, n_classes)
        P = predicted_probs[idx]      # shape (Nc, n_classes)

        # Soft confusion matrix = sum over outer products
        # For each synapse: outer(true_onehot, predicted_probs)
        soft_conf = np.einsum('ni,nj->ij', T, P)

        # Normalize by number of synapses in cluster
        soft_conf /= len(idx)

        cluster_soft_confusion[c] = soft_conf

    return cluster_soft_confusion


#####################################
#            Accuracies             #
#####################################

# All Accuracies Boxplot
acc = dict(zip(alg_labels, np.array(accuracies)[:, 0]))

plt.figure(figsize=(10, 6))
plt.boxplot(acc.values())

# Customize plot
plt.xticks(ticks=range(1, 8), labels=acc.keys(), rotation=45)
plt.hlines(accuracies[0][1][0], xmin=0, xmax=8)
plt.ylabel('Accuracy')
plt.title('Classifier Performance Comparison')

# Show the plot
plt.tight_layout()
# plt.show()

# ADB Perfomance
perfo_ind = {0: "sensitivity", 1: "specificity", 2: "fpr", 3: "fnr"}

fig, axs = plt.subplots(nrows=2, ncols=2)
(ax1, ax2, ax3, ax4) = axs.flatten()

for i, cell in enumerate(color_dict.keys()):
    ax1.hist(np.array(perfo_outputs[2][i])[:, 0], color=color_dict[cell], alpha=0.6, label=cell, bins=20)
    # ax1.set_title(f"{perfo_ind[0]}")

    print(f"{cell} {perfo_ind[0]}: {np.nanmean(np.array(perfo_outputs[2][i])[:, 0])}")

    ax2.hist(np.array(perfo_outputs[2][i])[:, 1], color=color_dict[cell], alpha=0.6, label=cell, bins=20)
    # ax2.set_title(f"{perfo_ind[1]}")

    print(f"{cell} {perfo_ind[1]}: {np.nanmean(np.array(perfo_outputs[2][i])[:, 1])}")

    ax3.hist(np.array(perfo_outputs[2][i])[:, 2], color=color_dict[cell], alpha=0.6, label=cell, bins=20)
    # ax3.set_title(f"{perfo_ind[2]}")

    print(f"{cell} {perfo_ind[2]}: {np.nanmean(np.array(perfo_outputs[2][i])[:, 2])}")

    ax4.hist(np.array(perfo_outputs[2][i])[:, 3], color=color_dict[cell], alpha=0.6, label=cell, bins=20)
    # ax4.set_title(f"{perfo_ind[3]}")

    print(f"{cell} {perfo_ind[3]}: {np.nanmean(np.array(perfo_outputs[2][i])[:, 3])}")

    ax1.tick_params(labelsize=6)
    ax2.tick_params(labelsize=6)
    ax3.tick_params(labelsize=6)
    ax4.tick_params(labelsize=6)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)

fig.set_dpi(1200)
fig.tight_layout()
# fig.set_size_inches(1.522, 1.548)
# fig.set_size_inches(1.918, 1.759)
fig.set_size_inches(1.418, 1.301)

plt.savefig(f"./Figures/fig3/adb_performance.svg", transparent=True)

# plt.show()

#####################################
#             Continuum             #
#####################################

srp_fits = []
cell_types = []

directory = "./Data/srp_fits_in_rodent/"

all_cell_ids = []
for i in range(1, len(os.listdir(directory))):
    file_name = os.listdir(directory)[i]
    cell_type = file_name.split('_')[0]
    cell_id = file_name.split('_')[3:]
    cell_id = f"{cell_id[0]}_{cell_id[1]}_{cell_id[2].split('.p')[0]}"
    key = (cell_type, cell_id)
    pickle_file = open(directory + file_name, "rb")
    params1 = pickle.load(pickle_file)
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus = params1
    sigma_scale = [1] # 4
    mu_scale = None
    constructed_params = (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )

    if sigma_baseline == 6 and sigma_amps == 1000:
        continue

    if sigma_baseline == None:
        continue

    fitted_params = [mu_baseline, *mu_amps, *mu_taus, sigma_baseline, sigma_amps, *sigma_taus]
    # fitted_params = [mu_baseline, *mu_amps, *mu_taus]
    srp_fits.append(fitted_params)
    cell_types.append(cell_type)
    all_cell_ids.append(cell_id)

prob_dict = {key: np.mean(np.stack(arr_list), axis=0)
             for key, arr_list in final_prob_dict.items()}

cell_id_prob_dict = {all_cell_ids[key]: np.mean(np.stack(arr_list), axis=0)
             for key, arr_list in final_prob_dict.items()}

# PCA
pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())
scaler.fit(np.array(list(prob_dict.values())))
scaled_arr = scaler.transform(np.array(list(prob_dict.values())))
pca_result = pca.fit_transform(scaled_arr)

pc1 = pca_result[:, 0]
pc2 = pca_result[:, 1]

# Blend colors based on probabilities
blended_colors = []
keys = list(prob_dict.keys())
for key in keys:
    probs = prob_dict[key]
    blended = probs[0] * color1 + probs[1] * color2 + probs[2] * color3
    # if probs == [0, 0, 1]:
    #     blended = probs[0] * color1 + probs[1] * color2 + probs[2] * color3
    # else:
    #     blended = 'white'
    blended_colors.append(blended)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# labels = kmeans.fit_predict(pca_result[:, :2])
labels = kmeans.fit_predict(list(prob_dict.values()))

# Continuum
clus_pca = {f"arm{i + 1}" : pca_result[labels == i] for i in range(n_clusters)}
clus_fits = {f"arm{i + 1}" : np.array(srp_fits)[labels == i] for i in range(n_clusters)}
clus_colors = {f"arm{i + 1}" : np.array(blended_colors)[labels == i] for i in range(n_clusters)}
clus_prob = {f"arm{i + 1}" : np.array(list(prob_dict.values()))[labels == i] for i in range(n_clusters)}
clus_cell_types = {f"arm{i + 1}" : np.array(cell_types)[labels == i] for i in range(n_clusters)}

# Plotting
colors = [color1, color2, color3]

fig, ax = plt.subplots()

ax.scatter(pc1, pc2, c=blended_colors, alpha=0.5, s=8, edgecolor='none')

for i in range(n_clusters):
    ax.scatter(np.mean(clus_pca[f"arm{i + 1}"][:, 0]), np.mean(clus_pca[f"arm{i + 1}"][:, 1]), c='black', s=20)

cmap = mcolors.LinearSegmentedColormap.from_list("pvalb_sst_vip", colors)
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['pvalb', 'sst', 'vip'])
cbar.ax.tick_params(labelsize=6)

# ax.title("Arms 1 and 2", fontsize=14)
# ax.xlabel("PC1")
# ax.ylabel("PC2")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=6)
fig.set_dpi(1200)
fig.set_size_inches(4.734, 3.153)
# plt.grid(True)
# plt.tight_layout()
plt.savefig("./Figures/fig4/GMM_continuum.svg", transparent=True)
# plt.show()

from collections import Counter

index_dict = {'pvalb': 0,
              'sst': 1,
              'vip': 2}

fig, ax = plt.subplots()
list_prob = list(prob_dict.values())

predicted_num_in_clus = {}
for i in range(n_clusters):
    mean = np.mean(clus_prob[f"arm{i + 1}"], axis=0)
    predicted_num_in_clus[i] = mean * len(clus_prob[f"arm{i + 1}"])

ax.scatter(np.array(list_prob)[:, 0], np.array(list_prob)[:, 2], color=blended_colors, s=8, alpha=0.5, edgecolor='none')
ax.plot([0, 1], [1, 0], color='black', linestyle='--')

for i in range(n_clusters):

    clus_array = clus_fits[f"arm{i + 1}"][:, :5]

    min_vals = np.nanmin(clus_array, axis=0)
    max_vals = np.nanmax(clus_array, axis=0)

    clus_array_norm = (clus_array - min_vals) / (max_vals - min_vals)

    model_medoid = KMedoids(n_clusters=1, metric='euclidean').fit(clus_array_norm)
    prob_medoid = clus_prob[f"arm{i + 1}"][:, :5][model_medoid.medoid_indices_[0]]

    ax.scatter(prob_medoid[0], prob_medoid[2], c='black', s=20)

# ax.plot(xs, ys, 'r-', linewidth=2, label="Isomap curve")
# ax.scatter(prob_arrays[:,0], prob_arrays[:,1], c=X_1d, cmap='viridis', s=10)

cmap = mcolors.LinearSegmentedColormap.from_list("pvalb_sst_vip", colors)
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, shrink=0.75)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['pvalb', 'sst', 'vip'])
cbar.ax.tick_params(labelsize=6)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=6)
fig.set_dpi(1200)
# fig.set_size_inches(4.934, 1.772)
# fig.set_size_inches(3.787, 1.367) # paper
fig.set_size_inches(5.714, 2.468)
# plt.grid(True)
# plt.tight_layout()
plt.savefig("./Figures/fig4/Continuum_pv_vip_axes.svg", transparent=True)

plt.show()

#####################################
#    True distribution continuum    #
#####################################

prob_arrays = []
clus_cell_arrays = []
for i in clus_prob.keys():
    for k, j in enumerate(clus_prob[i]):
        prob_arrays.append([j[0], j[2]])
        clus_cell_arrays.append(clus_cell_types[i][k])
prob_arrays = np.array(prob_arrays)
clus_cell_arrays = np.array(clus_cell_arrays)

iso = Isomap(n_neighbors=10, n_components=1)
X_1d = iso.fit_transform(prob_arrays).ravel()

fig, ax = plt.subplots()

ax.hist(np.array(X_1d)[np.where(clus_cell_arrays == 'pvalb')[0]], bins=50, alpha=0.5, color=colors[0], density=True)
ax.hist(np.array(X_1d)[np.where(clus_cell_arrays == 'sst')[0]], bins=50, alpha=0.5, color=colors[1], density=True)
ax.hist(np.array(X_1d)[np.where(clus_cell_arrays == 'vip')[0]], bins=50, alpha=0.5, color=colors[2], density=True)
ax.invert_xaxis()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=4)
fig.set_dpi(1200)
fig.set_size_inches(1.386, 0.685) # paper# plt.grid(True)
# plt.tight_layout()
plt.savefig("./Figures/fig4/Pre_type_distribution_along_Continuum.svg", transparent=True)

plt.show()

#####################################
#        Params Histograms          #
#####################################

params_dict = {
    0: ["mu_baseline", 0],
    1: ["mu_amp1", 1],
    2: ["mu_amp2", 2],
    3: ["mu_amp3", 3],
    4: ["mu_amp4", 4],
    5: ["sigma_baseline", 9],
    6: ["sigma_amp", 10]
}

for j in range(len(params_dict.keys())):

    fig, ax = plt.subplots()

    for i, cell in enumerate(color_dict.keys()):
        dist = np.array(srp_fits)[np.array(cell_types) == cell]
        dist = dist[:, params_dict[j][1]]
        dist = dist[dist != None]
        dist = dist[dist < 50000]

        ax.hist(dist, bins=20, alpha=0.5, label=cell, color=color_dict[cell], density=True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6)

    fig.set_dpi(1200)
    # fig.set_size_inches(0.865, 0.845)
    # fig.set_size_inches(0.583, 0.704)
    # fig.set_size_inches(0.796, 0.607) # fig 4 - model
    fig.set_size_inches(0.991, 0.62)  # supp - phy
    fig.savefig(f"./Figures/fig4/Hist_{params_dict[j][0]}.svg", transparent=True)
    # plt.show()

#####################################
#              Kernels              #
#####################################

# --- Cluster Medoid Kernels ---
for i in range(len(clus_fits)):
    fig, ax = plt.subplots()

    clus_array = clus_fits[f"arm{i + 1}"][:, 0:5] # 0:5 for means, 9:11 for sigmas

    min_vals = np.nanmin(clus_array, axis=0)
    max_vals = np.nanmax(clus_array, axis=0)

    clus_array_norm = (clus_array - min_vals) / (max_vals - min_vals)

    model_medoid = KMedoids(n_clusters=1, metric='euclidean').fit(clus_array_norm)
    mean_fit = clus_array[model_medoid.medoid_indices_[0]]

    # mean_fit = np.mean(clus_fits[f"arm{i + 1}"], axis=0)

    params = {"mu_baseline": mean_fit[0],
              "mu_amps": mean_fit[1:5], # 1:5 for means, 1 for sigmas
              "mu_taus": [5, 15, 200, 4000], # [5, 15, 200, 4000] for means, 400 for sigmas
              "SD": None,
              "mu_scale": None}

    model = easySRP(**params)

    kernel_y = model.run_ISIvec([200, 401], fast=False, return_all=True)["filtered_spiketrain"][:6000]
    # kernel_x = np.arange(0, 2000, 0.1)[:10000] - 200 + i * 200  # Shift by 200 ms per kernel
    kernel_x = np.arange(0, 2000, 0.1)[:6000] - 200

    ax.plot(kernel_x, kernel_y, color=np.mean(clus_colors[f"arm{i + 1}"], axis=0), linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.tick_params(labelsize=4, width=0.3, length=1)
    ax.tick_params(labelsize=4, width=0.5, length=1.5)

    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    ax.locator_params(axis='x', nbins=2)
    # ax.locator_params(axis='y', nbins=2)

    # ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune=None, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune=None, steps=[1, 2, 5, 10]))

    fig.set_dpi(1200)
    # fig.set_size_inches(0.352, 0.265) # mean
    # fig.set_size_inches(0.269, 0.202)  # mean final
    # fig.set_size_inches(0.452, 0.345)  # mean final 2
    # fig.set_size_inches(0.561, 0.421)  # sigma
    # fig.set_size_inches(0.433, 0.332)  # sigma
    fig.set_size_inches(0.616, 0.468)  # sigma final 2

    plt.savefig(f"./Figures/fig4/GMM_kernel_mu{i + 1}.svg", transparent=True)

    # plt.show()

# --- Cell Type Medoid Kernels
fig, ax = plt.subplots(figsize=(8, 6))

error_dict = {'pvalb': -1, 'sst': -1, 'vip': -1}

for i, cell in enumerate(color_dict.keys()):

    cell_array = np.array(srp_fits)[np.array(cell_types) == cell][:, 0:5] # 0:5 for means, 9:11 for sigmas

    min_vals = np.nanmin(cell_array, axis=0)
    max_vals = np.nanmax(cell_array, axis=0)

    cell_array_norm = (cell_array - min_vals) / (max_vals - min_vals)

    model_medoid = KMedoids(n_clusters=1, metric='euclidean').fit(cell_array_norm)
    medoid = cell_array_norm[model_medoid.medoid_indices_[0]]

    dists = np.linalg.norm(cell_array_norm - medoid, axis=1)

    trim_fraction = 0.1
    k = int(len(cell_array_norm) * trim_fraction)

    farthest_idx = np.argsort(dists)[-k:]

    trimmed_array_norm = np.delete(cell_array_norm, farthest_idx, axis=0)
    trimmed_array = np.delete(cell_array, farthest_idx, axis=0)

    model_medoid = KMedoids(n_clusters=1, metric='euclidean').fit(trimmed_array_norm)
    mean_fit = trimmed_array[model_medoid.medoid_indices_[0]]

    all_1 = [1] * len(np.array(list(prob_dict.values()))[np.array(cell_types) == cell])
    error_dict[cell] = np.mean(np.array(all_1) - np.array(list(prob_dict.values()))[np.array(cell_types) == cell][:, i])

    params = {"mu_baseline": mean_fit[0],
              "mu_amps": mean_fit[1:5], # 1:5 for means, 1 for sigmas
              "mu_taus": [5, 15, 200, 4000], # [5, 15, 200, 4000] for means, 400 for sigmas
              "SD": None,
              "mu_scale": None}

    model = easySRP(**params)

    kernel_y = model.run_ISIvec([200, 401], fast=False, return_all=True)["filtered_spiketrain"][:6000]
    # kernel_x = np.arange(0, 2000, 0.1)[:10000] - 200 + i * 200  # Shift by 200 ms per kernel
    kernel_x = np.arange(0, 2000, 0.1)[:6000] - 200

    ax.plot(kernel_x, kernel_y, color=color_dict[cell], alpha=0.5, linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=6)

fig.set_dpi(1200)
# fig.set_size_inches(1, 0.772)
fig.set_size_inches(1.19, 0.966)
plt.savefig(f"./Figures/fig4/Pre_kernel_mus_medoid.svg", transparent=True)

# plt.show()

#####################################
#            Phys Hist              #
#####################################

params_dict = {
    0: "areas",
    1: "release_prob",
    2: "first_fifth",
    3: "first_second",
    4: "recovery_50"
}

data = pickle.load(open("./Data/processed_input_in_rodent.p", "rb"))
srp_fits = np.array(data)[:, 3:8] # 3:8 for phys params
cell_types = np.array(data)[:, 0]

for j in range(len(params_dict.keys())):

    fig, ax = plt.subplots()

    for i, cell in enumerate(color_dict.keys()):
        dist = np.array(srp_fits)[np.array(cell_types) == cell]
        dist = dist[:, j]
        dist = dist[dist != None]
        dist = dist[dist < 50000]

        ax.hist(dist, bins=20, alpha=0.5, label=cell, color=color_dict[cell], density=True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6)

    fig.set_dpi(1200)
    # fig.set_size_inches(0.865, 0.845)
    # fig.set_size_inches(0.583, 0.704)
    # fig.set_size_inches(0.796, 0.607) # fig 4 - model
    fig.set_size_inches(0.991, 0.62)  # supp - phy
    fig.savefig(f"./Figures/sup1/Hist_{params_dict[j]}.svg", transparent=True)
    # plt.show()