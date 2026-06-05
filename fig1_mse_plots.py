
"""
Runs the SRP model on stimulus trains, compares model predictions to data, computes error metrics (MSE),
organizes those errors by cell type, and generates boxplots summarizing model performance.

Authors:
    jgben — original implementation
    jadepoir — adaptations for inhibitory project (2026)
"""

import os
import math
import pickle
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from srplasticity.srp import ExpSRP
from srplasticity.tm import TsodyksMarkramModel

#----------------------------------------------------------------------------------

fig_folder = f"./Figures/fig1/"
in_or_ex = "in" # ex
num_bf = 4 # Number of base functions summed to construct efficacy kernels
#----------------------------------------------------------------------------------
  
#dictionary to capitalize cre-types  
cre_capital = {
    'pvalb':'Pvalb',
    'nr5a1':'Nr5a1',
    'sim1':'Sim1',
    'sst':'Sst',
    'vip':'Vip',
    'ntsr1':'Ntsr1',
    'tlx3':'Tlx3',
    'fam84b':'Fam84b',
    'rorb':'Rorb',
    'pvalb,sst':'Pvalb,Sst',
    'unknown':'Unknown',
    'tac1':'Tac1'
    }

#----------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------

markersize = 3
capsize = 2
lw = 1
def plot_mufit(axis, target_dict_20, target_dict_50, srp_mean):

    xax = np.arange(12)
    #print("testing arrays from fit: nanmean, nanstd for 20hz")
    #print(np.nanmean(target_dict_20, 0))
    #print(np.nanstd(target_dict_20, 0))

    #"""  
    axis.errorbar(
        xax,
        np.nanmean(target_dict_20, 0),
        yerr=stat.sem(target_dict_20, 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        #yerr=np.nanstd(target_dict_20, 0),
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
        #yerr=np.nanstd(target_dict_50, 0) / 2,
        #yerr=np.nanstd(target_dict_50, 0),
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

#----------------------------------------------------------------------------------

def plot_mufit2(axis, target_dict, srp_mean):

    xax = np.arange(12)

    color = {"tm": "#0077bb", "srp": "#fd3c06", "accents": "grey"}
    for key in srp_mean.keys():
        axis.errorbar(
            xax,
            np.nanmean(target_dict[key], 0),
            yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
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
    axis.set_ylim(0, 8)

#----------------------------------------------------------------------------------

def plot_mufit3(axis, target_dict, srp_mean, key):

    xax = np.arange(12)

    color = {"tm": "#0077bb", "srp": "#fd3c06", "accents": "grey"}

    axis.errorbar(
        xax,
        np.nanmean(target_dict[key], 0),
        yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
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

#----------------------------------------------------------------------------------

max_threshold = -0.005

pickle_file = open("./Data/Extracted_STP_1.3mM_in_Rodent.p", "rb")
recordings = pickle.load(pickle_file)

target_dict = {}
training_stim_dict = {}
pre_types = []
post_types = []

for type_pair in recordings.keys():
    type1, type2 = type_pair
    pre_types.append(type1)
    post_types.append(type2)

unique_pres = np.unique(np.asarray(pre_types))
unique_posts = np.unique(np.asarray(post_types))
pre_mses = {}
pre_sigma_mses = {}
pre_fits = {}
pre_sigma_fits = {}

for pre in unique_pres:
    pre_mses[pre] = []
    pre_sigma_mses[pre] = []
    pre_fits[pre] = {}
    pre_sigma_fits[pre] = {}

post_mses = {}
post_sigma_mses = {}
post_fits = {}
post_sigma_fits = {}
pair_mses = {}

for post in unique_posts:
    post_mses[post] = []
    post_sigma_mses[post] = []
    post_fits[post] = {}
    post_sigma_fits[post] = {}

for type_pair in recordings.keys():

    type1, type2 = type_pair
    chosen_dict = recordings[type_pair]

    for pair_id in chosen_dict.keys():
        if pair_id != 'pair_IDs':

            first_spike_list = []
            testing_counter = 0
            pair_mses[pair_id] = {}

            for protocol in chosen_dict[pair_id]:
                testing_counter += 1

                if not isinstance(protocol, int):
                    clamp, freq, delay = protocol

                    if freq == None or delay == None:
                        continue

                    print(protocol)

                    if clamp == 'ic':
                        pair_mses[pair_id][protocol[1:]] = []
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            divisor = chosen_dict[pair_id][protocol][i, 0]

                            if in_or_ex == 'in':
                                if divisor < -1E-9:
                                    if divisor > max_threshold:
                                        first_spike_list.append(divisor)
                                else:
                                    first_spike_list.append(-1E-9)
                            if in_or_ex == 'ex':
                                if divisor > 1E-9:
                                    if divisor < max_threshold:
                                        first_spike_list.append(divisor)
                                else:
                                    first_spike_list.append(1E-9)
            if len(first_spike_list) > 0:
                averaged_divisor = sum(first_spike_list)/len(first_spike_list)
                #apply normalisation
            else:
                print("voltage clamp runs only for pair")
            for protocol in chosen_dict[pair_id]:
                if not isinstance(protocol, int):
                    #print(protocol)
                    clamp, freq, delay = protocol

                    if freq == None or delay == None:
                        continue

                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            added_row = chosen_dict[pair_id][protocol][i,:]
                            safe = True
                            for n in range(0, len(added_row)):
                                if in_or_ex == 'in':
                                    if chosen_dict[pair_id][protocol][i, n] > -1E-9:
                                        added_row[n] = -1E-9
                                    if chosen_dict[pair_id][protocol][i, n] < max_threshold:
                                        safe = False
                                if in_or_ex == 'ex':
                                    if chosen_dict[pair_id][protocol][i, n] < 1E-9:
                                        added_row[n] = 1E-9
                                    if chosen_dict[pair_id][protocol][i, n] > max_threshold:
                                        safe = False
                            #skip row if values above threshold=
                            if not safe:
                                continue
                            normed_row = added_row/averaged_divisor

                            exceeding_bound = False
                            if in_or_ex == 'ex':
                                for response in normed_row:
                                    if response > 17.5:
                                        exceeding_bound = True
                                if pair_id == '<Pair 1502751709.407 6 5>':
                                    for response in normed_row:
                                        if response > 12:
                                            exceeding_bound = True
                            if in_or_ex == 'in':
                                for response in normed_row:
                                    if response > 7:
                                        exceeding_bound = True

                            if exceeding_bound:
                                continue
                            try:
                                target_dict[pair_id][(freq, delay)] = np.vstack((target_dict[pair_id][(freq, delay)], normed_row))
                            except:
                                try:
                                    target_dict[pair_id][(freq, delay)] = normed_row
                                except:
                                    target_dict[pair_id] = {(freq, delay): normed_row}
                                try:
                                    print(pair_id)
                                    print((freq, delay))
                                    training_stim_dict[pair_id][(freq, delay)] = [0] + [1000/freq] * 7 + [delay*1000] + [1000/freq]*3 #should be [0] + ...
                                    print(training_stim_dict[pair_id][(freq, delay)])
                                except:
                                    training_stim_dict[pair_id] = {(freq, delay): [0] + [1000/freq] * 7 + [delay*1000] + [1000/freq]*3}

#----------------------------------------------------------------------------------

pair_id_list = []
srp_means = {}
srp_sigmas = {}

directory = f"./Data/srp_fits_in_rodent"


for i in range(1, len(os.listdir(directory))):
    file_name = os.listdir(directory)[i]
    print(file_name)

    pair_id = file_name.split('_')[3:]
    pre_type= file_name.split('_')[0]
    post_type= file_name.split('_')[1]
    pair_id = f"{pair_id[0]}_{pair_id[1]}_{pair_id[2].replace('.p', '')}"
    print((pair_id, pre_type, post_type))
    pkl1_file = open(directory + "/" + file_name, "rb")
    params1 = pickle.load(pkl1_file)
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amp, sigma_tau = params1

    if sigma_amp == None:
        continue

    sigma_scale=[1]
    mu_scale=None
    constructed_params = (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amp,
        sigma_tau,
        mu_scale,
        sigma_scale,
    )

    pair_id_list.append((pair_id, pre_type, post_type))
    pair_id = pair_id.split("_")
    pair_id = f"<Pair {pair_id[0]} {pair_id[1]} {pair_id[2]}>"

    srp_mean1, srp_sigma1, srp_est1 = get_model_estimates(ExpSRP(*constructed_params), training_stim_dict[pair_id])
    srp_means[pair_id] = srp_mean1
    srp_sigmas[pair_id] = srp_sigma1

for pair_id, pre_type, post_type in pair_id_list:
    pair_id = pair_id.split("_")
    pair_id = f"<Pair {pair_id[0]} {pair_id[1]} {pair_id[2]}>"
    keys = target_dict[pair_id].keys()
    protocols = []
    for key in keys:
        if key != 'pair_ID':
            protocols.append(key)

    num_rows = int(math.ceil(len(protocols)/3))
    if len(protocols)>1:
        for index in range(0, len(protocols)):
            x = int(index/3)
            y = index % 3
            num_runs = len(target_dict[pair_id][protocols[index]])
            if target_dict[pair_id][protocols[index]].ndim < 2:
                continue
            title = "Protocol: " + str(protocols[index][0])+"Hz, "+ str(protocols[index][1])+"s, "+str(num_runs)+" runs"

            if len(target_dict[pair_id][protocols[index]]) > 1: #multiple runs case
                errors = [[] for i in range(0, 12)]
                model_vals = [[] for i in range(0, 12)]
                sigma_errors = [[] for i in range(0, 12)]
                model_sigma_vals = [[] for i in range(0, 12)]
                raw_vals = [[] for i in range(0, 12)]
                raw_sigma_vals = [[] for i in range(0, 12)]
                raw_vals_all = [[] for i in range(0, 12)]

                for i in range(0, 12):
                    errors[i] = np.square(np.nanmean(target_dict[pair_id][protocols[index]][:,i]) - np.nanmean(srp_means[pair_id][protocols[index]][i]))
                    model_vals[i] = np.nanmean(srp_means[pair_id][protocols[index]][i])
                    sigma_errors[i] = np.square(np.nanstd(target_dict[pair_id][protocols[index]][:,i]) - np.nanmean(srp_sigmas[pair_id][protocols[index]][i]))
                    model_sigma_vals[i] = np.nanmean(srp_sigmas[pair_id][protocols[index]][i])
                    raw_vals[i] = np.nanmean(target_dict[pair_id][protocols[index]][:,i])
                    raw_sigma_vals[i] = np.nanstd(target_dict[pair_id][protocols[index]][:,i])
                    raw_vals_all[i] = target_dict[pair_id][protocols[index]][:,i]

                errors = np.transpose(np.stack(errors))
                sigma_errors = np.transpose(np.stack(sigma_errors))

                if len(pre_mses[pre_type]) > 0:
                    pre_mses[pre_type] = np.append(pre_mses[pre_type], [errors], axis=0)
                    pre_sigma_mses[pre_type] = np.append(pre_sigma_mses[pre_type], [sigma_errors], axis=0)

                else:
                    pre_mses[pre_type] = np.asarray([errors])
                    pre_sigma_mses[pre_type] = np.asarray([sigma_errors])

                if len(pre_fits[pre_type]) > 0:
                    pre_fits[pre_type]["model"] = np.append(pre_fits[pre_type]["model"], [model_vals], axis=0)
                    pre_fits[pre_type]["model sigmas"] = np.append(pre_fits[pre_type]["model sigmas"], [model_sigma_vals], axis=0)
                    pre_fits[pre_type]["raw"] = np.append(pre_fits[pre_type]["raw"], [raw_vals], axis=0)
                    pre_fits[pre_type]["raw sigmas"] = np.append(pre_fits[pre_type]["raw"], [raw_sigma_vals], axis=0)
                    pre_fits[pre_type]["raw_all"] = np.concatenate((pre_fits[pre_type]["raw_all"], raw_vals_all), axis=1)
                else:
                    pre_fits[pre_type]["model"] = np.asarray([model_vals])
                    pre_fits[pre_type]["model sigmas"] = np.asarray([model_sigma_vals])
                    pre_fits[pre_type]["raw"] = np.asarray([raw_vals])
                    pre_fits[pre_type]["raw sigmas"] = np.asarray([raw_sigma_vals])
                    pre_fits[pre_type]["raw_all"] = np.asarray(raw_vals_all)

                if len(post_mses[post_type]) > 0:
                    post_mses[post_type] = np.append(post_mses[post_type], [errors], axis=0)
                    post_sigma_mses[post_type] = np.append(post_sigma_mses[post_type], [sigma_errors], axis=0)
                else:
                    post_mses[post_type] = np.asarray([errors])
                    post_sigma_mses[post_type] = np.asarray([sigma_errors])
                try:
                    if len(pair_mses[pair_id][protocols[index]]) > 0:
                        print(protocols[index])
                        pair_mses[pair_id][protocols[index]] = np.append(pair_mses[pair_id][protocols[index]], [errors], axis=0)
                    else:
                        pair_mses[pair_id][protocols[index]] = np.asarray([errors])
                except:
                    print(f'Failed to add errors to pair_mses for {pair_id}, {protocols[index]}')

                if len(post_fits[post_type]) > 0:
                    post_fits[post_type]["model"] = np.append(post_fits[post_type]["model"], [model_vals], axis=0)
                    post_fits[post_type]["model sigmas"] = np.append(post_fits[post_type]["model sigmas"], [model_sigma_vals], axis=0)
                    post_fits[post_type]["raw"] = np.append(post_fits[post_type]["raw"], [raw_vals], axis=0)
                    post_fits[post_type]["raw sigmas"] = np.append(post_fits[post_type]["raw sigmas"], [raw_sigma_vals], axis=0)
                    #post_fits[post_type]["raw_all"] = np.append(post_fits[post_type]["raw_all"], [raw_vals_all], axis=0)
                    post_fits[post_type]["raw_all"] = np.concatenate((post_fits[post_type]["raw_all"], raw_vals_all), axis=1)
                else:
                    post_fits[post_type]["model"] = np.asarray([model_vals])
                    post_fits[post_type]["model sigmas"] = np.asarray([model_sigma_vals])
                    post_fits[post_type]["raw"] = np.asarray([raw_vals])
                    post_fits[post_type]["raw sigmas"] = np.asarray([raw_sigma_vals])
                    post_fits[post_type]["raw_all"] = np.asarray(raw_vals_all)

print(pre_mses.keys())
print(pre_mses['ntsr1'])
print(pre_mses['pvalb,sim1'])
print(pre_mses['pvalb,sst'])
print(pre_mses['sim1'])
print(pre_mses['sim1,pvalb'])
print(pre_mses['tac1'])
print(pre_mses['unknown'])

with_great_mse = set()

largest_mse = None
largest_pair_id = None
largest_protocol = None
for pair_id in pair_mses:
    for protocol in pair_mses[pair_id]:
        mean = np.mean(pair_mses[pair_id][protocol])
        if ~np.isnan(mean):
            if mean > 5:
                print(pair_id)
                print(protocol)
                print(mean)
                print("---------------")
                with_great_mse.add(pair_id)
            if largest_mse == None:
                largest_mse = mean
                largest_pair_id = pair_id
                largest_protocol = protocol
            else:
                if mean > largest_mse:
                    largest_mse = mean
                    largest_pair_id = pair_id
                    largest_protocol = protocol
print("/////////////////")
print(largest_pair_id)
print(largest_protocol)
print(largest_mse)

print(with_great_mse)

#slice arrays by column to get distributions within each cre_type
post_2_vals = []
pre_2_vals = []
post_2_labels = []
pre_2_labels = []

post_8_vals = []
pre_8_vals = []
post_8_labels = []
pre_8_labels = []


post_9_vals = []
pre_9_vals = []
post_9_labels = []
pre_9_labels = []

post_all_vals = []
pre_all_vals = []
post_all_labels = []
pre_all_labels = []

post_first8_vals = []
pre_first8_vals = []
post_first8_labels = []
pre_first8_labels = []

post_first8_sigma_vals = []
pre_first8_sigma_vals = []

post_last4_vals = []
pre_last4_vals = []
post_last4_labels = []
pre_last4_labels = []

post_last4_sigma_vals = []
pre_last4_sigma_vals = []

for post_type in post_mses.keys():
    if len(post_mses[post_type]) > 1:
        #print(post_mses[post_type])

        data_2 = post_mses[post_type][:,1]
        post_2_vals.append(data_2[~np.isnan(data_2)]) #get all vals in col (stim) 2 that aren't nan
        post_2_labels.append(cre_capital[post_type])

        data_8 = post_mses[post_type][:,7]
        post_8_vals.append(data_8[~np.isnan(data_8)]) #get all vals in col (stim) 2
        post_8_labels.append(cre_capital[post_type])

        data_9 = post_mses[post_type][:,8]
        post_9_vals.append(data_9[~np.isnan(data_9)]) #get all vals in col (stim) 2
        post_9_labels.append(cre_capital[post_type])

        #compute overall post
        data_all = np.ndarray.flatten(post_mses[post_type][:,:])
        post_all_vals.append(data_all[~np.isnan(data_all)]) #get all vals in col (stim) 2
        post_all_labels.append(cre_capital[post_type])

        #compute first 8 post
        data_first8 = np.ndarray.flatten(post_mses[post_type][:,:8])
        post_first8_vals.append(data_first8[~np.isnan(data_first8)]) #get all vals in col (stim) 2
        data_sigma_first8 = np.ndarray.flatten(post_sigma_mses[post_type][:,:8])
        post_first8_sigma_vals.append(data_sigma_first8[~np.isnan(data_sigma_first8)])
        post_first8_labels.append(cre_capital[post_type])

        #compute overall post
        data_last4 = np.ndarray.flatten(post_mses[post_type][:,8:])
        post_last4_vals.append(data_last4[~np.isnan(data_last4)]) #get all vals in col (stim) 2
        data_sigma_last4 = np.ndarray.flatten(post_sigma_mses[post_type][:, 8:])
        post_last4_sigma_vals.append(data_sigma_last4[~np.isnan(data_sigma_last4)])
        post_last4_labels.append(cre_capital[post_type])

for pre_type in pre_mses.keys():
    if len(pre_mses[pre_type]) > 1:
        #print("pre_mses[pre_type]:")
        #print(pre_mses[pre_type])

        data_2 = pre_mses[pre_type][:,1]
        pre_2_vals.append(data_2[~np.isnan(data_2)]) #get all vals in col (stim) 2
        pre_2_labels.append(pre_type)

        data_8 = pre_mses[pre_type][:,7]
        pre_8_vals.append(data_8[~np.isnan(data_8)]) #get all vals in col (stim) 2
        pre_8_labels.append(pre_type)

        data_9 = pre_mses[pre_type][:,8]
        pre_9_vals.append(data_9[~np.isnan(data_9)]) #get all vals in col (stim) 2
        pre_9_labels.append(pre_type)

        #compute overall pre
        data_all = np.ndarray.flatten(pre_mses[pre_type][:,:])
        pre_all_vals.append(data_all[~np.isnan(data_all)]) #get all vals in col (stim) 2
        pre_all_labels.append(pre_type)

        #compute first 8
        data_first8 = np.ndarray.flatten(pre_mses[pre_type][:,:8])
        pre_first8_vals.append(data_first8[~np.isnan(data_first8)]) #get all vals in col (stim) 2
        data_sigma_first8 = np.ndarray.flatten(pre_sigma_mses[pre_type][:, :8])
        pre_first8_sigma_vals.append(data_sigma_first8[~np.isnan(data_sigma_first8)])
        pre_first8_labels.append(pre_type)

        #compute last 4
        data_last4 = np.ndarray.flatten(pre_mses[pre_type][:,8:])
        pre_last4_vals.append(data_last4[~np.isnan(data_last4)]) #get all vals in col (stim) 2
        data_sigma_last4 = np.ndarray.flatten(pre_sigma_mses[pre_type][:, 8:])
        pre_last4_sigma_vals.append(data_sigma_last4[~np.isnan(data_sigma_last4)])
        pre_last4_labels.append(pre_type)

save_dir = './Figures/fig1/'

#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_8_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_8_vals, medianprops=dict(color="xkcd:teal blue"))
plt.title("MSE of Model Fit to Eighth Simulation")
plt.xticks(x_coordinates, post_8_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_to_Eighth_Simulation.svg"
# f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_9_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_9_vals, medianprops=dict(color="xkcd:teal blue"))
plt.title("MSE of Model Fit to Ninth Simulation")
plt.xticks(x_coordinates, post_9_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_to_Ninth_Simulation.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()


#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_all_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_all_vals, medianprops=dict(color="xkcd:teal blue"))
plt.title("MSE of Fit by Post-Type")
plt.xticks(x_coordinates, post_all_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_of_Post_Fit.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(pre_all_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(pre_all_vals, medianprops=dict(color="xkcd:teal blue"))
plt.title("MSE of Fit by Pre-Type")
plt.xticks(x_coordinates, pre_all_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_of_Pre_Fit.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by first 8 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_first8_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_first8_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.title("Fit MSE of First 8 Responses by Post-Type")
plt.xticks(x_coordinates, post_first8_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_of_Post_Fit_first8.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by first 8 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_first8_sigma_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_first8_sigma_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.title("Fit MSE of First 8 SDs by Post-Type")
plt.xticks(x_coordinates, post_first8_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_of_Post_Fit_SD_first8.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by first 8 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(pre_first8_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6)
plt.boxplot(pre_first8_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.xticks(x_coordinates, pre_first8_labels)
savetitle = save_dir+"MSE_of_Pre_Fit_first8.svg"
f.set_size_inches((1.999, 1.149))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by first 8 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(pre_first8_sigma_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6)
plt.boxplot(pre_first8_sigma_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.xticks(x_coordinates, pre_first8_labels)
savetitle = save_dir+"MSE_of_Pre_Fit_SD_first8.svg"
f.set_size_inches((1.999, 1.149))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#print fit means
all_first8_vals = []
all_first8_sigma_vals = []
all_first8_vals_not_sst =[]
sst_first8_vals = []
for i in range(0, len(post_first8_vals)):
    print("first 8 vals")
    print("mean MSE responses for "+post_first8_labels[i]+" post = "+str(np.mean(np.asarray(post_first8_vals[i]))))
    print("SEM of mean MSE for " + post_first8_labels[i] + " post = " + str(stat.sem(np.asarray(post_first8_vals[i]))))
    print("mean MSE SDs for "+post_first8_labels[i]+" post = "+str(np.mean(np.asarray(post_first8_sigma_vals[i]))))
    print("SEM of mean MSE for " + post_first8_labels[i] + " post SD = " + str(stat.sem(np.asarray(post_first8_sigma_vals[i]))))
    all_first8_vals = all_first8_vals+[val for val in post_first8_vals[i]]
    all_first8_sigma_vals = all_first8_sigma_vals + [val for val in post_first8_sigma_vals[i]]
    if post_first8_labels[i] == 'Sst':
        print("adding sst")
        sst_first8_vals  = post_first8_vals[i]
    else:
        all_first8_vals_not_sst = all_first8_vals_not_sst+[val for val in post_first8_vals[i]]


print("testing sst first 8 t-test vs all")
print(ttest_ind(np.asarray(sst_first8_vals), np.asarray(all_first8_vals_not_sst)))

print("mean MSE across all first 8 responses: "+str(np.mean(np.asarray(all_first8_vals))))
print("SEM of mean MSE across all first 8: "+str(stat.sem(np.asarray(all_first8_vals))))
print("mean MSE across all first 8 SDs: "+str(np.mean(np.asarray(all_first8_sigma_vals))))
print("SEM of mean MSE across all first 8 SD: "+str(stat.sem(np.asarray(all_first8_sigma_vals))))

#boxplot distributions by last 4 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_last4_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_last4_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.title("Fit MSE of Last 4 by Post-Type")
plt.xticks(x_coordinates, post_last4_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_of_Post_Fit_last4.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by last 4 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_last4_sigma_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.boxplot(post_last4_sigma_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.title("Fit MSE of Last 4 SDs by Post-Type")
plt.xticks(x_coordinates, post_last4_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
savetitle = save_dir+"MSE_of_Post_Fit_SD_last4.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by last 4 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(pre_last4_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6)
plt.boxplot(pre_last4_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.xticks(x_coordinates, pre_last4_labels)
savetitle = save_dir+"MSE_of_Pre_Fit_last4.svg"
f.set_size_inches((1.999, 1.149))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#boxplot distributions by last 4 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(pre_last4_sigma_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6)
plt.boxplot(pre_last4_sigma_vals, showfliers=False, medianprops=dict(color="xkcd:teal blue"))
plt.xticks(x_coordinates, pre_last4_labels)
savetitle = save_dir+"MSE_of_Pre_Fit_SD_last4.svg"
f.set_size_inches((1.999, 1.149))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)
# plt.show()

#print fit means
all_last4_vals = []
all_last4_sigma_vals = []
all_last4_vals_not_sst =[]
sst_last4_vals = []
for i in range(0, len(post_last4_vals)):
    print("Last 4 vals")
    print("mean MSE responses for "+post_last4_labels[i]+" post = "+str(np.mean(np.asarray(post_last4_vals[i]))))
    print("SEM of mean MSE for " + post_last4_labels[i] + " post = " + str(stat.sem(np.asarray(post_last4_vals[i]))))
    print("mean MSE SDs for "+post_last4_labels[i]+" post = "+str(np.mean(np.asarray(post_last4_sigma_vals[i]))))
    print("SEM of mean MSE for " + post_last4_labels[i] + " post SD = " + str(stat.sem(np.asarray(post_last4_sigma_vals[i]))))
    all_last4_vals = all_last4_vals+[val for val in post_last4_vals[i]]
    all_last4_sigma_vals = all_last4_sigma_vals + [val for val in post_last4_sigma_vals[i]]
    if post_last4_labels[i] == 'Sst':
        print("adding sst")
        sst_last4_vals  = post_last4_vals[i]
    else:
        all_last4_vals_not_sst = all_last4_vals_not_sst+[val for val in post_last4_vals[i]]

print("mean MSE across all last 4 responses: "+str(np.mean(np.asarray(all_last4_vals))))
print("SEM of mean MSE across all last 4: "+str(stat.sem(np.asarray(all_last4_vals))))
print("mean MSE across all last 4 SDs: "+str(np.mean(np.asarray(all_last4_sigma_vals))))
print("SEM of mean MSE across all last 4 SD: "+str(stat.sem(np.asarray(all_last4_sigma_vals))))

#--------------------------------------------------
#plot model and mse by stim for each cre_type

for post_type in post_mses.keys():
    if len(post_mses[post_type]) > 1: #normally 4
        model_vals = []
        model_sigma_vals = []
        raw_vals = []
        raw_sds = []
        raw_sems = []
        for i in range(0, 12):
            model_data = post_fits[post_type]["model"][:, i]
            mean = np.nanmean(model_data)
            model_sigma_data = post_fits[post_type]["model sigmas"][:, i]
            sigma = np.nanmean(model_sigma_data)
            model_vals.append(mean)
            model_sigma_vals.append(sigma)

            raw_data = post_fits[post_type]["raw"][:, i]
            raw_data_sigma = post_fits[post_type]["raw sigmas"][:, i]
            raw_data_all = post_fits[post_type]["raw_all"][:, i]
            raw_mean = np.nanmean(raw_data)
            raw_mean_sd = np.nanmean(raw_data_sigma)
            raw_vals.append(raw_mean)
            raw_sds.append(raw_mean_sd)
            raw_sem = stat.sem(raw_data_all, axis=None, ddof=0, nan_policy='omit')
            raw_sems.append(raw_sem)
        x_coordinates = [i for i in range(1, len(model_vals)+1)]
        x_coordinates_1 = [i for i in range(1, 9)]
        x_coordinates_2 = [i for i in range(9, 13)]
        f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
        f.set_size_inches((3.15, 1.97))
        f.set_dpi(600)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.set_visible(False)

        ax.errorbar(x_coordinates, raw_vals, yerr=raw_sds, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7)
        ax2.errorbar(x_coordinates, raw_vals, yerr=raw_sds, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7)

        ax.scatter(x_coordinates, raw_vals, color="black")
        ax2.scatter(x_coordinates, raw_vals, color="black")

        ax.plot(x_coordinates, model_vals, label="Model", color="#fd3c06")
        ax2.plot(x_coordinates, model_vals, label="Model", color="#fd3c06")

        ax.fill_between(x_coordinates, [a_i - b_i for a_i, b_i in zip(model_vals, model_sigma_vals)], [a_i + b_i for a_i, b_i in zip(model_vals, model_sigma_vals)], color="#fd3c06", alpha=0.2)
        ax2.fill_between(x_coordinates, [a_i - b_i for a_i, b_i in zip(model_vals, model_sigma_vals)], [a_i + b_i for a_i, b_i in zip(model_vals, model_sigma_vals)], color="#fd3c06", alpha=0.2)

        #set limits to divide subplots over delay
        ax.set_xlim(0.5,8.01)
        ax2.set_xlim(8.98,12.5)

        #add diagonal lines to delay break
        d = .01 # how big to make the diagonal lines in axes coordinates initially 0.015
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-(d/2),1+(d/2)), (-d,+d), linewidth=3, **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d+0.01,+d+0.01), (-d,+d), linewidth=3, **kwargs)

        f.suptitle(cre_capital[post_type]+" Post")

        ax.set_xticks(x_coordinates_1)
        ax2.set_xticks(x_coordinates_2)

        '''
        if post_type == "sst":
            pickel_file = fig_folder+"sst_plot_means"
            with open(pickel_file, 'wb') as handle:
                pass
        save_name = fig_folder+"model_data_fit_post_"+post_type+".svg"
        f.tight_layout()
        '''

for pre_type in pre_mses.keys():
    if len(pre_mses[pre_type]) > 1: #normally 4
        model_vals = []
        model_sigma_vals = []
        raw_vals = []
        raw_sds = []
        raw_sems = []
        for i in range(0, 12):
            model_data = pre_fits[pre_type]["model"][:, i]
            mean = np.nanmean(model_data)
            model_vals.append(mean)

            model_sigma_data = pre_fits[pre_type]["model sigmas"][:, i]
            mean_sigma = np.nanmean(model_sigma_data)
            model_sigma_vals.append(mean_sigma)

            raw_data = pre_fits[pre_type]["raw"][:, i]
            raw_sigma_data = pre_fits[pre_type]["raw sigmas"][:, i]
            raw_data_all = pre_fits[pre_type]["raw_all"][:, i]
            raw_mean = np.nanmean(raw_data)
            raw_mean_sd = np.nanmean(raw_sigma_data)
            raw_sd = np.nanstd(raw_data)
            #raw_sem = stat.sem(raw_data, axis=None, ddof=0, nan_policy='omit')
            raw_sem = stat.sem(raw_data_all, axis=None, ddof=0, nan_policy='omit')
            raw_sems.append(raw_sem)
            raw_vals.append(raw_mean)
            raw_sds.append(raw_mean_sd)

        x_coordinates = [i for i in range(1, len(model_vals)+1)]
        x_coordinates_1 = [i for i in range(1, 9)]
        x_coordinates_2 = [i for i in range(9, 13)]

        f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
        f.set_size_inches((1.910, 1.152))
        f.set_dpi(1200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.set_visible(False)

        ax.errorbar(x_coordinates, raw_vals, yerr=raw_sds, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7)
        ax2.errorbar(x_coordinates, raw_vals, yerr=raw_sds, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7)#changed to sem

        ax.scatter(x_coordinates, raw_vals, color="black", s=15)
        ax2.scatter(x_coordinates, raw_vals, color="black", marker="o", s=15)

        ax.plot(x_coordinates, model_vals, label="Model", color="#fd3c06")
        ax2.plot(x_coordinates, model_vals, label="Model", color="#fd3c06")

        ax.fill_between(x_coordinates, [a_i - b_i for a_i, b_i in zip(model_vals, model_sigma_vals)],
                        [a_i + b_i for a_i, b_i in zip(model_vals, model_sigma_vals)], color="#fd3c06", alpha=0.4)
        ax2.fill_between(x_coordinates, [a_i - b_i for a_i, b_i in zip(model_vals, model_sigma_vals)],
                         [a_i + b_i for a_i, b_i in zip(model_vals, model_sigma_vals)], color="#fd3c06", alpha=0.4)

        #set limits to divide subplots over delay
        ax.set_xlim(0.5,8.05)
        ax2.set_xlim(8.94,12.5)

        #add diagonal lines to delay break
        d = .01 # how big to make the diagonal lines in axes coordinates initially 0.015
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-(d/2),1+(d/2)), (-d,+d), linewidth=3, **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d+0.01,+d+0.01), (-d,+d), linewidth=3, **kwargs)

        ax.set_xticks(x_coordinates_1)
        ax2.set_xticks(x_coordinates_2)

        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)

        #plt.ylim([0,1])
        print(pre_type)
        save_name = fig_folder+"model_data_fit_pre_"+pre_type+".svg"
        f.tight_layout()
        plt.savefig(save_name, transparent=True)
        # plt.show()