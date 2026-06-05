
"""
Loads short‑term plasticity measures and synaptic response model fits for inhibitory mouse connections, merges
them into a unified dataset with physiological and model‑derived features, organizes the data by pre‑ and postsynaptic
cell type, and saves the processed output to a pickle file.

Authors:
    jgben — original implementation
    jadepoir — modifications for inhibitory project (2026)
"""

import os
import pickle
import numpy as np

#----------------------------------------------------------------------------------

#load in code for dictionaries
row_info = []
data = []
pyr_types = ['nr5a1', 'rorb']
pyr_indices = []

l5et_types = ['sim1', 'fam84b']
l5et_indices = []

pre_type_to_keep = "all"

pre_type_dict = {"pvalb": "PV", "vip": "VIP", "sst": "SST", "all" : "ALL"}

data_post_pre = {}
unique_pre = set()
unique_post = set()
unique_pairs = set()

measures_file = open('.Data/Measures_1.3mM_in_Rodent.p', "rb")
measures_dict = pickle.load(measures_file)

num_100s = 0
count = 0
directory = "./Data/srp_fits_in_rodent"
for i in range(0, len(os.listdir(directory))):
    file_name = os.listdir(directory)[i]
    print(file_name)
    count += 1
    print(count)
    pair_id = file_name.split('_')[3:]
    pre_type = file_name.split('_')[0]
    print(pre_type)
    post_type = file_name.split('_')[1]

    pair_id = f"{pair_id[0]}_{pair_id[1]}_{pair_id[2].replace('.p', '')}"

    pickle_file = open(directory + "/" + file_name, "rb")
    params = pickle.load(pickle_file)
    print(params)
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amp, sigma_tau, sigma_scale = params
    new_row = []
    #add identifiers
    new_row.append(pre_type) #1
    new_row.append(post_type) #2
    new_row.append(pair_id) #3

    unique_pre.add(pre_type)
    unique_post.add(post_type)
    unique_pairs.add(pre_type+post_type)

    #add measures
    # try:
    key_50hz = ('ic', 50.0, 0.25)
    pair_id_2 = f"<Pair {pair_id.split('_')[0]} {pair_id.split('_')[1]} {pair_id.split('_')[2]}>"

    ppr = measures_dict[(pre_type, post_type)][pair_id_2]['Paired_Pulse_50Hz']

    if ppr != None:
        if ppr > 4: #1000 for the excessive value, note: this was set to 4 until March 9th when changed for some testing, 4 may still be the best value
            print("skipper ppr = "+str(ppr))
            continue

    try:
        areas = measures_dict[(pre_type, post_type)][pair_id_2]['areas_50hz_mean']
    except:
        print("Missing areas_50hz_mean")
        areas = None
    try:
        release_prob = measures_dict[(pre_type, post_type)][pair_id_2]['release_prob_all']
    except:
        print("Missing release_prob_all")
        release_prob = None
    try:
        first_fifth = measures_dict[(pre_type, post_type)][pair_id_2]['first_fifth_50hz_mean']
    except:
        print("Missing first_fifth_50hz_mean")
        first_fifth = None
    try:
        first_second = measures_dict[(pre_type, post_type)][pair_id_2]['first_second_50hz_mean']
    except:
        print("Missing first_second_50hz_mean")
        first_second = None
    try:
        recovery_50 = measures_dict[(pre_type, post_type)][pair_id_2][key_50hz]['recovery']
    except:
        print(f"Missing {key_50hz}")
        recovery_50 = None
    #new_row.append(ppr)
    new_row.append(areas)
    new_row.append(release_prob)
    new_row.append(first_fifth)
    new_row.append(first_second)
    new_row.append(recovery_50)


    #add model params
    new_row.append(mu_baseline)
    new_row = new_row + [mu_amp for mu_amp in mu_amps]
    new_row.append(sigma_baseline)
    new_row = new_row + [sigma_amp]

    data.append(new_row)

    if post_type in data_post_pre:
        if pre_type in data_post_pre[post_type]:
            curr_data = data_post_pre[post_type][pre_type]
            data_post_pre[post_type][pre_type] = np.vstack((curr_data, np.array(new_row)))
        else:
             data_post_pre[post_type][pre_type] = np.array(new_row)
    else:
        data_post_pre[post_type] = {pre_type: np.array(new_row)}

print(data)

num_cols = len(data[0])
params_data = [row[8:] for row in data[:]] #model only
phys_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
hybrid_data = [row[3:] for row in data[:]] #model and phys
model_data = [row[8:] for row in data[:]] #model only

# phys_labels = ["areas", "release_prob", "STP induction", "PPR", "50Hz Recovery"]

print("printing model data[0]")
print(model_data[0])

row_labels = [row[0:3] for row in data[:]]
params_arr = np.array(params_data)

#--------------------------------------------------------------------------

#rodent output pickle name
filename = f"data/processed_input_in_rodent.p"
with open(filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)