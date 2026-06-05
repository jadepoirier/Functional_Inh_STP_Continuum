
"""
Performs a statistical test of whether the clustering structure observed in real synaptic SRP parameters
is stronger than what would be expected from clustering random noise.

Author: jadepoir
"""

import copy
import pickle
import random
import numpy as np
from sklearn import metrics
from collections import Counter
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())
fits = pickle.load(open("./Data/processed_input_in_rodent.p", "rb"))
# fits = pickle.load(open("./Data/processed_input_in_human.p", "rb"))
# fits = pickle.load(open("./Data/processed_input_ex_rodent.p", "rb"))

# srp_fits_pretype = [[pair[0]] + pair[3:8] for pair in fits if None not in pair[3:8]] # in rodent/human phys params
# srp_fits_without_pretype = [pair[3:8] for pair in fits if None not in pair[3:8]] # in rodent/human phys params
srp_fits_pretype = [[pair[0]] + pair[8:-2] for pair in fits if None not in pair[8:-2]] # in rodent/human model params
srp_fits_without_pretype = [pair[8:-2] for pair in fits if None not in pair[8:-2]] # in rodent/human model params
# srp_fits_pretype = [[pair[0]] + pair[8:-1] for pair in fits if None not in pair[8:-1]] # ex rodent model params
# srp_fits_without_pretype = [pair[8:-1] for pair in fits if None not in pair[8:-1]] # ex rodent model params

num_params = len(srp_fits_pretype[0]) - 1
mean = [0] * num_params

pv_count = len([i[0] for i in srp_fits_pretype if i[0] == 'pvalb'])
sst_count = len([i[0] for i in srp_fits_pretype if i[0] == 'sst'])
vip_count = len([i[0] for i in srp_fits_pretype if i[0] == 'vip'])

true_sc = []
art_sc = []

true_min = []
art_min = []

true_num_clusters = []
art_num_clusters = []
art_dicts_SCs = []
true_dicts_SCs = []
true_reachability = []
art_reachability = []

# scaler.fit(np.array(srp_fits_without_pretype))
# scaled_arr_full = scaler.transform(np.array(srp_fits_without_pretype))
# true_pca_result_full = pca.fit_transform(scaled_arr_full)

count = 0
for i in range(3000):
    # with in rodent
    srp_fits = random.sample(list(np.array(srp_fits_pretype)[:, 1:][:pv_count]), k=vip_count)
    srp_fits += random.sample(list(np.array(srp_fits_pretype)[:, 1:][pv_count:pv_count + sst_count]), k=vip_count)
    srp_fits += list(np.array(srp_fits_pretype)[:, 1:][pv_count + sst_count:])

    # with ex rodent and in human
    # srp_fits = random.sample(list(np.array(srp_fits_without_pretype)), k=round(len(srp_fits_without_pretype) * 0.8)) # with ex rodent or in human
    # srp_fits = list(np.array(srp_fits_without_pretype))

    # print(len(srp_fits))

    scaler.fit(np.array(srp_fits))
    scaled_arr = scaler.transform(np.array(srp_fits))
    true_pca_result = pca.fit_transform(scaled_arr)

    # true_pca_result = true_pca_result_full
    # true_pca_result = random.sample(list(np.array(true_pca_result_full)[:pv_count]), k=vip_count)
    # true_pca_result += random.sample(list(np.array(true_pca_result_full)[pv_count:pv_count + sst_count]), k=vip_count)
    # true_pca_result += random.sample(list(np.array(true_pca_result_full)[pv_count + sst_count:]), k=vip_count)
    # # true_pca_result += list(np.array(true_pca_result_full)[pv_count + sst_count:])

    # plt.scatter(pca_result[:, 0][:574], pca_result[:, 1][:574], color="xkcd:dark blue")
    # plt.scatter(pca_result[:, 0][574:574 + 297], pca_result[:, 1][574:574 + 297], color="xkcd:red orange")
    # plt.scatter(pca_result[:, 0][574 + 297:], pca_result[:, 1][574 + 297:], color="xkcd:golden yellow")
    # plt.show()

    # gm = GaussianMixture(n_components=num_params, random_state=None).fit(np.array(true_pca_result)[:, :])
    # artificial_data = gm.sample(n_samples=len(true_pca_result))[0]
    #
    # scaler.fit(np.array(artificial_data))
    # scaled_arr = scaler.transform(np.array(artificial_data))
    # art_pca_result = pca.fit_transform(scaled_arr)

    min_cluster_sizes = [i for i in range(2, len(true_pca_result))]  # specify desired list of minimum cluster sizes

    true_sc = -10
    art_sc = -10

    num_clusters_true = -1
    num_clusters_art = -1

    best_true_clustering = []
    best_art_clustering = []

    true_real_clus_size = 0
    art_real_clus_size = 0

    for min_size in min_cluster_sizes:
        true_clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(true_pca_result)
        try:
            silhouette = metrics.silhouette_score(true_pca_result, true_clustering.labels_)
            num_clusters = len(Counter(true_clustering.labels_).keys())
            real_clus_size = Counter(true_clustering.labels_)[0]

            if (num_clusters == 2
                    and real_clus_size > true_real_clus_size):
                true_sc = silhouette
                num_clusters_true = num_clusters
                best_true_clustering = true_clustering
                true_real_clus_size = real_clus_size

        except:
            continue

    if true_sc == -10:
        continue

    print(f"True SC: {true_sc}")
    print(f"Num clusters: {num_clusters_true}")

    var = pca.explained_variance_
    other_var = [(sum(var) - var[0]) / (num_params - 1)] * (num_params - 1)
    d = np.diag([var[0]] + other_var)

    artificial_data = np.random.multivariate_normal(mean, d, len(true_pca_result))

    scaler.fit(np.array(artificial_data))
    scaled_arr = scaler.transform(np.array(artificial_data))
    art_pca_result = pca.fit_transform(scaled_arr)

    for min_size in min_cluster_sizes:
        art_clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(art_pca_result)

        try:
            silhouette = metrics.silhouette_score(art_pca_result, art_clustering.labels_)
            num_clusters = len(Counter(art_clustering.labels_).keys())
            real_clus_size = Counter(art_clustering.labels_)[0]

            if (num_clusters == 2
                    and real_clus_size > art_real_clus_size):
                art_sc = silhouette
                num_clusters_art = num_clusters
                best_art_clustering = art_clustering
                art_real_clus_size = real_clus_size
        except:
            continue

    if art_sc == -10:
        continue

    print(f"Art SC: {art_sc}")
    print(f"Num clusters: {num_clusters_art}")
    print("###################################")

    try:
        scs = list(pickle.load(open("./Data/sc_in_rodent_2clus_3000.p", "rb")))
        scs.append((true_sc, num_clusters_true, art_sc, num_clusters_art))
        if len(scs) > 3000:
            break
        else:
            pickle.dump(scs, open("./Data/sc_in_rodent_2clus_3000.p", "wb"))
            print(f"Count: {len(scs)}")

    except:
        pickle.dump([(true_sc, art_sc)], open("./Data/sc_in_rodent_2clus_3000.p", "wb"))
