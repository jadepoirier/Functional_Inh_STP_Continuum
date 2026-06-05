
"""
Loads fitted SRP model parameters and corresponding experimental STP data, generates model‑vs‑data
comparison plots for each synaptic pair across stimulation protocols, and saves the resulting figures.

Authors: jgben, jadepoir
"""

import os
import math
import pickle
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from srplasticity.srp import ExpSRP
from srplasticity.tm import TsodyksMarkramModel

# ----------------------------------------------------------------------------------

# dictionary mapping cre-types to capital
cre_capital = {
    'pvalb': 'Pvalb',
    'nr5a1': 'Nr5a1',
    'sim1': 'Sim1',
    'sst': 'Sst',
    'vip': 'Vip',
    'ntsr1': 'Ntsr1',
    'tlx3': 'Tlx3',
    'fam84b': 'Fam84b',
    'rorb': 'Rorb',
    'unknown': 'Unknown'
}


# ----------------------------------------------------------------------------------

def get_model_estimates(model, stimulus_dict):
    """
    :return: Model estimates for training dataset
    """
    estimates = {}
    if isinstance(model, ExpSRP):
        means = {}
        sigmas = {}
        for key, isivec in stimulus_dict.items():
            means[key], sigmas[key], estimates[key] = model.run_ISIvec(
                isivec, ntrials=10000
            )
        return means, sigmas, estimates

    elif isinstance(model, TsodyksMarkramModel):
        for key, isivec in stimulus_dict.items():
            estimates[key] = model.run_ISIvec(isivec)
            model.reset()

        return estimates

    else:
        for key, isivec in stimulus_dict.items():
            estimates[key] = model.run_ISIvec(isivec)

        return estimates


# ----------------------------------------------------------------------------------

markersize = 3
capsize = 2
lw = 1


def plot_mufit(axis, target_dict_20, target_dict_50, srp_mean):
    xax = np.arange(12)
    print("testing arrays from fit: nanmean, nanstd for 20hz")
    # print(np.nanmean(target_dict_20, 0))
    # print(np.nanstd(target_dict_20, 0))

    # """
    axis.errorbar(
        xax,
        np.nanmean(target_dict_20, 0),
        yerr=stat.sem(target_dict_20, 0, nan_policy='omit'),
        # yerr=np.nanstd(target_dict_20, 0) / 2,
        # yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label="20 Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )

    axis.errorbar(
        xax,
        np.nanmean(target_dict_50, 0),
        yerr=stat.sem(target_dict_50, 0, nan_policy='omit'),
        # yerr=np.nanstd(target_dict_50, 0) / 2,
        # yerr=np.nanstd(target_dict_50, 0),
        color="black",
        marker="s",
        label="50 Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )

    color = {"tm": "#0077bb", "srp": "#fd3c06", "accents": "grey"}
    axis.plot(srp_mean["20"], color=color["srp"], ls="dashed", zorder=10)
    axis.plot(srp_mean["50"], color=color["srp"], zorder=10)

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    axis.set_ylim(0, 8)

    axis.legend(frameon=False)


# ----------------------------------------------------------------------------------

def plot_mufit2(axis, target_dict, srp_mean):
    xax = np.arange(12)
    print("testing arrays from fit: nanmean, nanstd for 20hz")

    # """
    color = {"tm": "#0077bb", "srp": "#fd3c06", "accents": "grey"}
    for key in srp_mean.keys():
        axis.errorbar(
            xax,
            np.nanmean(target_dict[key], 0),
            yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
            # yerr=np.nanstd(target_dict_20, 0) / 2,
            # yerr=np.nanstd(target_dict_20, 0),
            color="black",
            ls="dashed",
            marker="o",
            label="" + str(key) + " Hz",
            capsize=capsize,
            elinewidth=0.7,
            lw=lw,
            markersize=markersize,
        )
        axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    axis.set_ylim(0, 8)

    # axis.legend(frameon=False)


# ----------------------------------------------------------------------------------

def plot_mufit3(axis, target_dict, srp_mean, srp_sigma, key):
    xax = np.arange(12)

    # """
    color = {"tm": "#0077bb", "srp": "#fd3c06", "accents": "grey"}
    # for key in srp_mean.keys():
    axis.errorbar(
        xax,
        np.nanmean(target_dict[key], 0),
        # np.nanmedian(target_dict[key], 0),
        yerr=np.nanstd(target_dict[key], 0, nan_policy='omit'),
        # yerr=np.nanstd(target_dict_20, 0) / 2,
        # yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label="" + str(key) + " Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)
    axis.fill_between(xax, [ai - bi for ai, bi in zip(srp_mean[key], srp_sigma[key])],
                      [ai + bi for ai, bi in zip(srp_mean[key], srp_sigma[key])], color=color["srp"],
                      ls="dashed", zorder=10)

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")


# ----------------------------------------------------------------------------------

def plot_mufit4(f, ax, ax2, target_dict, srp_mean, key):
    xax = np.arange(12)

    # """
    """
    color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
    #for key in srp_mean.keys():
    axis.errorbar(
        xax,
        np.nanmean(target_dict[key], 0),
        #np.nanmedian(target_dict[key], 0),
        yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        #yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label=""+str(key)+" Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)


    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    #axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])

    #axis.legend(frameon=False)
    """

    x_coordinates = [i for i in range(1, 13)]
    x_coordinates_1 = [i for i in range(1, 9)]
    x_coordinates_2 = [i for i in range(9, 13)]
    # plt.figure()
    # ax = plt.subplot(111)
    # f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
    # f.set_size_inches((3.15, 1.97))
    f.set_size_inches((1.965, 1.152))
    f.set_dpi(600)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    # plt.bar(x_coordinates, y_vals, yerr=y_errors)
    # plt.plot(x_coordinates, model_vals, label="Model")

    print("printing raw_sems")
    print(stat.sem(target_dict[key], 0, nan_policy='omit')[0:8])
    print("printing raw vals")
    print(np.nanmean(target_dict[key], 0)[0:8])
    # plt.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data") #changed to sem
    ax.errorbar(x_coordinates, np.nanmean(target_dict[key], 0), yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
                label="Data", color="black",
                ls="dashed", capsize=capsize, elinewidth=0.7)
    ax2.errorbar(x_coordinates, np.nanmean(target_dict[key], 0), yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
                 label="Data", color="black",
                 ls="dashed", capsize=capsize, elinewidth=0.7)  # changed to sem

    ax.scatter(x_coordinates, np.nanmean(target_dict[key], 0), color="black", marker="o", s=15)
    ax2.scatter(x_coordinates, np.nanmean(target_dict[key], 0), color="black", marker="o", s=15)

    ax.plot(x_coordinates, srp_mean[key], label="Model", color="#fd3c06")
    ax2.plot(x_coordinates, srp_mean[key], label="Model", color="#fd3c06")

    # set limits to divide subplots over delay
    ax.set_xlim(0.5, 8.01)
    ax2.set_xlim(8.98, 12.5)

    # add diagonal lines to delay break
    d = .01  # how big to make the diagonal lines in axes coordinates initially 0.015
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - (d / 2), 1 + (d / 2)), (-d, +d), linewidth=3, **kwargs)
    # ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d + 0.01, +d + 0.01), (-d, +d), linewidth=3, **kwargs)

    # plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
    # f.suptitle("Model Fit For "+pre_type +" Pre_type", fontsize =16)
    # ax.set_title("Mean Value of Model and Data by Stimulus for "+post_type+" Post_type")
    # plt.legend()
    # ax2.legend()

    """
    plt.xticks(x_coordinates, x_coordinates)
    plt.xlabel("Stimulus #")
    plt.ylabel("Normalized Amplitude")
    """

    ax.set_xticks(x_coordinates_1)
    ax2.set_xticks(x_coordinates_2)

    ax.tick_params(labelsize=6)
    ax2.tick_params(labelsize=6)


def plot_mufit_sigma(f, ax, ax2, target_dict, srp_mean, srp_sigma, key):
    xax = np.arange(12)
    # print("testing arrays from fit: nanmean, nanstd for 20hz")
    # print(np.nanmean(target_dict_20, 0))
    # print(np.nanstd(target_dict_20, 0))
    # print(target_dict)
    # print(np.nanmean(target_dict[key], 0))

    # """
    """
    color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
    #for key in srp_mean.keys():
    axis.errorbar(
        xax,
        np.nanmean(target_dict[key], 0),
        #np.nanmedian(target_dict[key], 0),
        yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        #yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label=""+str(key)+" Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)


    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    #axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])

    #axis.legend(frameon=False)
    """

    x_coordinates = [i for i in range(1, 13)]
    x_coordinates_1 = [i for i in range(1, 9)]
    x_coordinates_2 = [i for i in range(9, 13)]
    # plt.figure()
    # ax = plt.subplot(111)
    # f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
    # f.set_size_inches((3.15, 1.97))
    f.set_size_inches((1.965, 1.152))
    f.set_dpi(600)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    # plt.bar(x_coordinates, y_vals, yerr=y_errors)
    # plt.plot(x_coordinates, model_vals, label="Model")

    print("printing raw_sems")
    print(stat.sem(target_dict[key], 0, nan_policy='omit')[0:8])
    print("printing raw vals")
    print(np.nanmean(target_dict[key], 0)[0:8])

    ax.errorbar(x_coordinates, np.nanmean(target_dict[key], 0), yerr=np.nanstd(target_dict[key], axis=0),
                label="Data", color="black",
                ls="dashed", capsize=capsize, elinewidth=0.7)
    ax2.errorbar(x_coordinates, np.nanmean(target_dict[key], 0), yerr=np.nanstd(target_dict[key], axis=0),
                 label="Data", color="black",
                 ls="dashed", capsize=capsize, elinewidth=0.7)  # changed to sem

    ax.scatter(x_coordinates, np.nanmean(target_dict[key], 0), color="black", marker="o", s=15)
    ax2.scatter(x_coordinates, np.nanmean(target_dict[key], 0), color="black", marker="o", s=15)

    ax.plot(x_coordinates, srp_mean[key], label="Model", color="#fd3c06")
    ax2.plot(x_coordinates, srp_mean[key], label="Model", color="#fd3c06")

    ax.fill_between(x_coordinates, [ai - bi for ai, bi in zip(srp_mean[key], srp_sigma[key])],
                    [ai + bi for ai, bi in zip(srp_mean[key], srp_sigma[key])], color="#fd3c06", alpha=0.2)
    ax2.fill_between(x_coordinates, [ai - bi for ai, bi in zip(srp_mean[key], srp_sigma[key])],
                     [ai + bi for ai, bi in zip(srp_mean[key], srp_sigma[key])], color="#fd3c06", alpha=0.2)

    # set limits to divide subplots over delay
    ax.set_xlim(0.5, 8.01)
    ax2.set_xlim(8.98, 12.5)

    # add diagonal lines to delay break
    d = .01  # how big to make the diagonal lines in axes coordinates initially 0.015
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - (d / 2), 1 + (d / 2)), (-d, +d), linewidth=3, **kwargs)
    # ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d + 0.01, +d + 0.01), (-d, +d), linewidth=3, **kwargs)

    # plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
    # f.suptitle("Model Fit For "+pre_type +" Pre_type", fontsize =16)
    # ax.set_title("Mean Value of Model and Data by Stimulus for "+post_type+" Post_type")
    # plt.legend()
    # ax2.legend()

    """
    plt.xticks(x_coordinates, x_coordinates)
    plt.xlabel("Stimulus #")
    plt.ylabel("Normalized Amplitude")
    """

    ax.set_xticks(x_coordinates_1)
    ax2.set_xticks(x_coordinates_2)

    ax.tick_params(labelsize=6)
    ax2.tick_params(labelsize=6)


# ----------------------------------------------------------------------------------

pre_type = 'nr5a1'
post_type = 'vip'
chosen_pair = 77886
max_threshold = -0.005

pickle_file = open("./Data/Extracted_STP_1.3mM_in_Rodent.p", "rb")
recordings = pickle.load(pickle_file)
target_dict = {}
training_stim_dict = {}

for type_pair in recordings.keys():
    type1, type2 = type_pair

    chosen_dict = recordings[type_pair]

    for pair_id in chosen_dict.keys():
        if pair_id != 'pair_IDs':
            first_spike_list = []
            testing_counter = 0

            for protocol in chosen_dict[pair_id]:
                testing_counter += 1
                if not isinstance(protocol, int):
                    clamp, freq, delay = protocol
                    print(protocol)
                    # first_spike_list = []

                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            divisor = chosen_dict[pair_id][protocol][i, 0]
                            # print("divisor = "+str(divisor))
                            # if divisor > 1E-6:
                            if divisor < -1E-9:
                                if divisor > max_threshold:
                                    first_spike_list.append(divisor)
                            else:
                                first_spike_list.append(-1E-9)
            if len(first_spike_list) > 0:
                averaged_divisor = sum(first_spike_list) / len(first_spike_list)
            else:
                print("voltage clamp runs only for pair")
            for protocol in chosen_dict[pair_id]:
                if not isinstance(protocol, int):
                    # print(protocol)
                    clamp, freq, delay = protocol
                    if freq == None or delay == None:
                        continue
                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            added_row = chosen_dict[pair_id][protocol][i, :]
                            safe = True
                            for n in range(0, len(added_row)):
                                """
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                                """
                                if chosen_dict[pair_id][protocol][i, n] > -1E-9:
                                    added_row[n] = -1E-9
                                if chosen_dict[pair_id][protocol][i, n] < max_threshold:
                                    safe = False
                            if not safe:
                                continue
                            normed_row = added_row / averaged_divisor
                            try:
                                target_dict[pair_id][(freq, delay)] = np.vstack(
                                    (target_dict[pair_id][(freq, delay)], normed_row))
                            except:
                                try:
                                    target_dict[pair_id][(freq, delay)] = normed_row
                                except:
                                    target_dict[pair_id] = {(freq, delay): normed_row}
                                try:
                                    training_stim_dict[pair_id][(freq, delay)] = [0] + [1000 / freq] * 7 + [
                                        delay * 1000] + [1000 / freq] * 3
                                except:
                                    training_stim_dict[pair_id] = {
                                        (freq, delay): [0] + [1000 / freq] * 7 + [delay * 1000] + [1000 / freq] * 3}

# ----------------------------------------------------------------------------------

pair_id_list = []
type_list = []
post_cut = pre_type + "_" + post_type + "_"
dir_name = "./Figures/fig1/plot_individual_fits/"

srp_means = {}
srp_sigmas = {}
directory = "./Data/srp_fits_in_rodent/"

for i in range(1, len(os.listdir(directory))):
    cut_on = '_'
    file_name = os.listdir(directory)[i]
    pair_id = f"{file_name.split('_')[3]}_{file_name.split('_')[4]}_{file_name.split('_')[5].split('.')[0]}"
    pre_type = file_name.split(cut_on)[0]
    post_type = file_name.split(cut_on)[1]
    type_list.append((pre_type, post_type))

    pickle_file = open(directory + "/" + file_name, "rb")
    params = pickle.load(pickle_file)
    print(params)

    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus = params
    sigma_scale = [1]
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

    pair_id_2 = f"<Pair {pair_id.split('_')[0]} {pair_id.split('_')[1]} {pair_id.split('_')[2]}>"
    pair_id_list.append(pair_id_2)

    srp_mean1, srp_sigma1, srp_est1 = get_model_estimates(ExpSRP(*constructed_params), training_stim_dict[pair_id_2])
    srp_means[pair_id_2] = srp_mean1
    srp_sigmas[pair_id_2] = srp_sigma1

for i in range(0, len(pair_id_list)):
    pair_id = pair_id_list[i]
    pre_type, post_type = type_list[i]
    keys = target_dict[pair_id].keys()
    protocols = []
    for key in keys:
        print(key)
        if key != 'pair_ID':
            protocols.append(key)

    num_rows = int(math.ceil(len(protocols) / 3))
    for index in range(0, len(protocols)):
        print("pair_ID: " + str(pair_id))
        x = int(index / 3)
        y = index % 3
        print("len(protocols) = " + str(len(protocols)))
        print("x=" + str(x) + " y=" + str(y) + " key =" + str(protocols[index]))
        num_runs = len(target_dict[pair_id][protocols[index]])
        if target_dict[pair_id][protocols[index]].ndim < 2:
            print("Dimension too low")
            continue
        plottitle = f"{cre_capital[pre_type]} to {cre_capital[post_type]} {str(protocols[index][0])} Hz {str(protocols[index][1])}s"
        savetitle = str(protocols[index][0]) + "Hz, " + "_" + str(protocols[index][1]) + "s" + "_" + str(
            num_runs) + " runs"
        f = plt.figure()
        ax = plt.axes()
        f.set_size_inches((3.15, 1.97))
        f.set_dpi(1200)
        f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
        try:
            plot_mufit_sigma(f, ax, ax2, target_dict[pair_id], srp_means[pair_id], srp_sigmas[pair_id],
                             protocols[index])
            f.set_size_inches((1.965, 1.152))
            f.set_dpi(600)
            f.tight_layout()
            pair_id_3 = f"{pair_id.split(' ')[1]}_{pair_id.split(' ')[2]}_{pair_id.split(' ')[3][0]}"
            save_name = dir_name + pre_type + "_" + post_type + "_" + savetitle + "_" + str(pair_id_3) + ".svg"

            plt.savefig(save_name, transparent=True)
        except:
            print("failed plot")
            continue
