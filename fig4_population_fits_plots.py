
"""
Plots predicted STP curves by cell type.

Author: jadepoir
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from srplasticity.srp import ExpSRP

recordings = pickle.load(open("./Data/Extracted_STP_1.3mM_in_Rodent.p", "rb"))

max_threshold = -0.005
target_dict = {}
training_stim_dict = {}

for type_pair in recordings.keys():
    type1, type2 = type_pair

    # if type1 != "vip":
    #     continue

    if type2 == "unknown":
        continue

    if type_pair not in target_dict:
        target_dict[type_pair] = {}

    if type_pair not in training_stim_dict:
        training_stim_dict[type_pair] = {}

    chosen_dict = recordings[type_pair]  # should be sst

    for pair_id in chosen_dict.keys():
        pair_id_2 = pair_id.replace(" ", "_")
        pair_id_2 = pair_id_2.replace("<", "")
        pair_id_2 = pair_id_2.replace(">", "")

        if pair_id != 'pair_IDs':

            # modify to work by pairs
            first_spike_list = []
            testing_counter = 0
            for protocol in chosen_dict[pair_id]:
                testing_counter += 1
                if not isinstance(protocol, int):
                    clamp, freq, delay = protocol

                    if freq == None or delay == None:
                        continue
                    # first_spike_list = []
                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            # changed to make average excluding rows with any column values below 1E-9
                            """
                            safe_row = True
                            for n in range(0, 8):
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                            """
                            divisor = chosen_dict[pair_id][protocol][i, 0]
                            # print("divisor = "+str(divisor))
                            # if divisor > 1E-6:
                            if divisor < -1E-9:
                                if divisor > max_threshold:
                                    first_spike_list.append(divisor)
                            else:
                                first_spike_list.append(-1E-9)
                        # if len(first_spike_list) > 0:
            # print(testing_counter)
            # print(protocol)
            if len(first_spike_list) > 0:
                averaged_divisor = sum(first_spike_list) / len(first_spike_list)
                # apply normalisation
            else:
                # print("voltage clamp runs only for pair")
                pass
            for protocol in chosen_dict[pair_id]:
                # for i in range(0, len(first_spike_list)):
                # divisor = chosen_dict[key][i, 0]
                # print("divisor = "+str(divisor))
                if not isinstance(protocol, int):
                    # print(protocol)
                    clamp, freq, delay = protocol

                    if freq == None or delay == None:
                        continue

                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            added_row = chosen_dict[pair_id][protocol][i, :]
                            valid_row = True
                            for n in range(0, len(added_row)):
                                """
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                                """
                                if chosen_dict[pair_id][protocol][i, n] > -1E-9:
                                    added_row[n] = -1E-9
                                if chosen_dict[pair_id][protocol][i, n] < max_threshold:
                                    valid_row = False
                            if not valid_row:
                                continue
                            # if divisor > 1E-5:
                            normed_row = added_row / averaged_divisor
                            # print(normed_row)

                            exceeding_norm = False
                            for response in normed_row:
                                if response > 7:
                                    exceeding_norm = True

                            if exceeding_norm == True:
                                continue

                            try:
                                target_dict[type_pair][pair_id][(freq, delay)] = np.vstack(
                                    (target_dict[type_pair][pair_id][(freq, delay)], normed_row))
                            except:
                                # target_dict[str(int(freq))] = chosen_dict[key][:, 0:8]
                                # print(freq)
                                try:
                                    target_dict[type_pair][pair_id][(freq, delay)] = normed_row
                                except:
                                    target_dict[type_pair][pair_id] = {(freq, delay): normed_row}
                                # print(int(freq))
                                try:
                                    training_stim_dict[type_pair][pair_id][(freq, delay)] = [0] + [1000 / freq] * 7 + [
                                        delay * 1000] + [1000 / freq] * 3  # should be [0] + ...
                                except:
                                    training_stim_dict[type_pair][pair_id] = {
                                        (freq, delay): [0] + [1000 / freq] * 7 + [delay * 1000] + [1000 / freq] * 3}

directory = "./Data/srp_fits_in_rodent"

srp_params = {}

all_target = {}
all_target_sigma = {}
all_means = {}
all_sigmas = {}

for i in os.listdir(directory):
    pre_type = i.split("_")[0]
    post_type = i.split("_")[1]

    if (pre_type, post_type) not in srp_params.keys():
        srp_params[(pre_type, post_type)] = {}

    # pair_id = f"{i.split('_')[3]}_{i.split('_')[4]}_{i.split('_')[5].split('.')[0]}"
    pair_id = f"<Pair {i.split('_')[3]} {i.split('_')[4]} {i.split('_')[5].split('.')[0]}>"

    srp_params[(pre_type, post_type)][pair_id] = pickle.load(open(directory + "/" + i, "rb"))

for type_pair in target_dict:
    pre_type, post_type = type_pair

    if pre_type not in all_target.keys():
        all_target[pre_type] = []
        all_target_sigma[pre_type] = []
        # all_target_all[pre_type] = [[] for i in range(12)]
        all_means[pre_type] = []
        all_sigmas[pre_type] = []

    for pair_id in target_dict[type_pair]:
        print(pair_id)

        if pair_id == "pair_IDs":
            continue

        try:
            params = srp_params[type_pair][pair_id]
            mu_baseline = params[0]
            mu_amps = params[1]
            mu_taus = params[2]
            sigma_baseline = params[3]
            sigma_amps = params[4]
            sigma_taus = params[5]
            mu_scale = None
            sigma_scale = 1
        except:
            continue

        if pre_type == 'vip':
            for i in range(2, 9):
                try:
                    pair_id_2 = pair_id.split(">")[0] + f"({i})" + ">"
                    print(pair_id_2)
                    params = srp_params[type_pair][pair_id_2]
                    mu_baseline = params[0]
                    mu_amps = params[1]
                    mu_taus = params[2]
                    sigma_baseline = params[3]
                    sigma_amps = params[4]
                    sigma_taus = params[5]
                    mu_scale = None
                    sigma_scale = 1
                except:
                    continue

        model = ExpSRP(mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus, mu_scale, sigma_scale)

        for protocol in target_dict[type_pair][pair_id]:
            means, sigmas, _ = model.run_ISIvec(training_stim_dict[type_pair][pair_id][protocol])

            all_means[pre_type].append(means)
            all_sigmas[pre_type].append(sigmas)

            if type(target_dict[type_pair][pair_id][protocol][0]) == np.float64:
                # plt.plot(target_dict[type_pair][pair_id][protocol])
                all_target[pre_type].append(target_dict[type_pair][pair_id][protocol])
                # for i in range(12):
                #     all_target_all[pre_type][i].append(target_dict[type_pair][pair_id][protocol][i])
            else:
                run = []
                run_sigma = []
                for i in range(12):
                    run.append(np.nanmean(target_dict[type_pair][pair_id][protocol][:,i], axis=0))
                    run_sigma.append(np.nanstd(target_dict[type_pair][pair_id][protocol][:, i], axis=0))
                    # for j in target_dict[type_pair][pair_id][protocol][:,i]:
                    #     all_target_all[pre_type][i].append(j)
                all_target[pre_type].append(run)
                all_target_sigma[pre_type].append(run_sigma)
                # for run in target_dict[type_pair][pair_id][protocol]:
                #     # plt.plot(list(range(1, 13)), run, color="grey", linewidth=0.8)
                #     all_target[pre_type].append(run)

            sigmas = 3 * sigmas

            # plt.errorbar(x=list(range(1,13)), y=means, yerr=sigmas, color="red")

            #plt.show()

total_sigmas_types = {}
total_means_types = {}

for type_pair in target_dict.keys():
    total_sigmas_types[type_pair[0]] = []
    for pair_id in target_dict[type_pair]:
        for protocol in target_dict[type_pair][pair_id]:
            if type(target_dict[type_pair][pair_id][protocol][0]) == np.float64:
                total_sigmas_types[type_pair[0]].append(target_dict[type_pair][pair_id][protocol])
            else:
                for run in target_dict[type_pair][pair_id][protocol]:
                    total_sigmas_types[type_pair[0]].append(run)

for pre_type in all_target.keys():
    print(pre_type)
    total_target = np.nanmean(all_target[pre_type], axis=0)
    total_target_sd = np.nanmean(all_target_sigma[pre_type], axis=0)
    sd_overall = np.nanstd(total_sigmas_types[pre_type], axis=0)
    # total_target_sem = stat.sem(all_target_all[pre_type], axis=None, ddof=0, nan_policy='omit')
    total_means = np.nanmean(all_means[pre_type], axis=0)
    total_sigmas = np.nanmean(all_sigmas[pre_type], axis=0)
    total_sd_sigmas = np.nanstd(all_sigmas[pre_type], axis=0)
    total_sd_means = np.nanstd(all_means[pre_type], axis=0)

    # raw_sems = []
    # for i in range(12):
    #     raw_data_all = all_target_all[pre_type][i]
    #     raw_sem = stat.sem(raw_data_all, axis=None, ddof=0, nan_policy='omit')
    #     raw_sems.append(raw_sem)

    print(f"Target: {total_target}")
    # print(f"SEM: {raw_sems}")
    print(f"Means: {total_means}")
    print(f"Sigmas: {total_sigmas}")

    total_sigmas_types[pre_type] = (total_sigmas, total_sd_sigmas)
    total_means_types[pre_type] = (total_means, total_sd_means)

color_types = {"pvalb": "xkcd:dark blue",
               "sst": "xkcd:orange red",
               "vip": "xkcd:golden yellow"
                }

all_predicted = all_means # all_means, all_sigmas
cell_class = 'pvalb' # pvalb, sst, vip

for pre_type in all_predicted: # ***
    if pre_type != cell_class: # ***
        continue
    fig, axs = plt.subplots()
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)

    ax.set_xlim(0.5, 8.01)
    ax2.set_xlim(8.98, 12.5)
    ax.set_ylim(0, 3)
    ax2.set_ylim(0, 3)
    ax.set_xticks(list(range(1, 9)), fontsize=0.5)
    ax2.set_xticks(list(range(9, 13)), fontsize=0.5)
    # ax.set_yticks(list(range(4)), fontsize=0.5)
    # ax2.set_yticks(list(range(4)), fontsize=0.5)

    d = .01
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - (d / 2), 1 + (d / 2)), (-d, +d), **kwargs) # linewidth=3
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d + 0.01, +d + 0.01), (-d, +d), **kwargs)

    for run in all_predicted[pre_type]: # ***
        ax.plot(list(range(1, 13)), run, color=color_types[pre_type], alpha=0.1, linewidth=0.2)
        ax2.plot(list(range(1, 13)), run, color=color_types[pre_type], alpha=0.1, linewidth=0.2)

ax.tick_params(labelsize=5)
ax2.tick_params(labelsize=5)
f.set_size_inches((1.234, 1.044))
f.set_dpi(1200)
plt.savefig(f"./Figures/fig4/all_means_{cell_class}.svg", transparent=True)
# plt.show()