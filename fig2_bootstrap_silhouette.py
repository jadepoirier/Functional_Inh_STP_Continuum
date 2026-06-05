
"""
Performs unsupervised clustering of SRP model parameters to discover natural groupings
of synaptic dynamics based on their short‑term plasticity properties.

Authors: jgben, jadepoir
"""

import copy
import pickle
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())

# fits = pickle.load(open("./Data/processed_input_in_human.p", "rb"))
# fits = pickle.load(open("./Data/processed_input_ex_rodent.p", "rb"))
fits = pickle.load(open("./Data/processed_input_in_rodent.p", "rb"))

locations = pickle.load(open("./Data/in_pairs_loc_all.p", "rb"))

srp_fits = [pair[8:-2] for pair in fits] # in rodent/human
# srp_fits = [pair[8:] for pair in fits] # ex rodent

# By pre-type
# srp_fits = [pair[8:-2] for pair in fits if pair[0] == 'pvalb']

# print(Counter([pair[1] for pair in locations]))

# # By cortical layer
# locations_dict = {pair[0]: pair[1] for pair in locations}
#
# srp_fits = []
# for pair in fits:
#     try:
#         if locations_dict[f"<Pair {pair[2].split('_')[0]} {pair[2].split('_')[1]} {pair[2].split('_')[2].split('.')[0]}>"] == "2/3":
#             srp_fits.append(pair[8:-2])
#         else:
#             continue
#     except:
#         print(f"{pair[2]} not in locations_dict")

scaler.fit(srp_fits)
scaled_arr = scaler.transform(srp_fits)
pca_result = pca.fit_transform(scaled_arr)

count = 0

min_cluster_sizes = [i for i in range(2, 90)]  # specify desired list of minimum cluster sizes
rodent_silhouette_scores = []
clusterings = []
for min_size in min_cluster_sizes:

    # # For ex rodent
    # if min_size != 8:
    #     continue

    clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(pca_result)
    # clustering = KMeans(n_clusters=min_size).fit(pca_model_result_rodent)
    count += 1
    print(count)

    try:
        # # labels = clustering.labels_
        # # pca_result_2 = np.array(pca_result[:])[np.where(np.array(labels) != -1)[0]]
        # # labels = labels[labels != -1]
        # all_scs = np.array(metrics.silhouette_samples(pca_result, clustering.labels_))
        # labels = np.array(clustering.labels_)
        # SCs = [np.nanmean(all_scs[np.where(labels == i)[0]]) for i in set(clustering.labels_) if i != -1]
        # rodent_silhouette_scores.append(np.nanmean(SCs))
        rodent_silhouette_scores.append(metrics.silhouette_score(pca_result, clustering.labels_)) # original SC computation
        clusterings.append(clustering.labels_)
    except:
        continue

max_sc = max(rodent_silhouette_scores)
opt_min_size = rodent_silhouette_scores.index(max(rodent_silhouette_scores)) + 2
labels = clusterings[rodent_silhouette_scores.index(max(rodent_silhouette_scores))]
print(labels)

print(f"Max silhouette coeff: {max_sc}")
print(f"Optimal min size: {opt_min_size}")

colour_set = ['xkcd:cobalt', 'xkcd:blood red', 'xkcd:pumpkin', 'xkcd:apple green', 'xkcd:barney purple', 'xkcd:brown']

fig, ax = plt.subplots()

for j, i in enumerate(sorted(list(set(labels)))):
    occurance_pc1 = list(np.array(pca_result[:, 0])[np.where(np.array(labels) == i)[0]])
    occurance_pc2 = list(np.array(pca_result[:, 1])[np.where(np.array(labels) == i)[0]])

    if i != -1:
        ax.scatter(occurance_pc1, occurance_pc2, color=colour_set[j], alpha=0.5, s=8, edgecolor='none', label=f"Cluster {j}")
    else:
        ax.scatter(occurance_pc1, occurance_pc2, color=colour_set[j], alpha=0.5, s=8, edgecolor='none')

    ax.tick_params(labelsize=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# ax.legend(fontsize=6)
fig.set_dpi(1200)
fig.set_size_inches(1.545, 1.497) # in rodent
# fig.set_size_inches(0.417, 0.525) # ex rodent
plt.tight_layout()
plt.savefig("./Figures/fig2/clustering_in_rodent.svg", transparent=True)