
"""
Loads synaptic pulse‑response data from the aisynphys database, extracts and quality‑controls amplitudes,
organizes them by synapse type, computes summary measures of short‑term synaptic dynamics, and saves the
resulting metrics to a pickle file.

Authors:
    jgben — original implementation
    jadepoir — modifications for inhibitory project (2026)
"""

import math
import pandas
import pickle
from aisynphys.dynamics import *
from neuroanalysis.data import TSeries
from aisynphys.database import SynphysDatabase

# WARNING: DOWNLOADS THE FULL 180 GB DATASET
db = SynphysDatabase.load_sqlite("D:/datasets/database/synphys_r2.1_full.sqlite", readonly=True)

# Load all synapses associated with mouse V1 projects
pairs = db.pair_query(
    synapse=True,
    species='mouse',
    synapse_type='in',
    acsf='1.3mM Ca & 1mM Mg',  # '1.3mM Ca & 1mM Mg' or '2mM Ca & Mg'
    electrical=False  # Exclude gap junctions
).all()

results = pandas.DataFrame(columns=['pair_id',
                                    'pre_cell',
                                    'post_cell',
                                    'rec_id',
                                    'clamp_mode',
                                    'ind_freq',
                                    'rec_delay',
                                    'amps',
                                    'psc_amp',
                                    'psp_amp']
                           )
# create set of cre types for later print out
unique_cre_types = set()

pair_ids_dict = {}

count = 0
pre_amps = {}
synapses_with_at_least_12 = []
for ix in range(len(pairs)):
    print('processing pair {} of {}'.format(ix + 1, len(pairs)))
    pair = pairs[ix]
    pair_ids_dict[pair.id] = str(pair)
    # print(pair_ids_dict)

    pr_recs = pulse_response_query(pair, db=db, data=True, spike_data=True).all()  # extract all response trains

    # cull out all PRs that didn't get a fit
    # pr_recs = [pr_rec for pr_rec in pr_recs if pr_rec.PulseResponseFit.fit_amp is not None]
    # sort by clamp mode and frequency
    sorted = sorted_pulse_responses(pr_recs)

    pre_amps[pair.id] = {}

    if sorted != {}:  # if any recordings have a fit

        # Loop through all recordings
        for key, recs in sorted.items():
            clamp_mode, ind_freq, rec_delay = key

            # only consider current clamp recordings
            if clamp_mode != 'ic':
                continue

            pre_amps[pair.id][(ind_freq, rec_delay)] = []

            for recording, pulses in recs.items():
                if 1 not in pulses or 2 not in pulses:
                    continue

                if len(pulses.keys()) != 12:
                    continue

                responses = []
                qc_pass_rec = True
                amp_crossed_upper_bound = False

                for pulse_n in pulses.keys():
                    # recordings for each pulse
                    rec = pulses[pulse_n]
                    qc_pass = rec.PulseResponse.in_qc_pass if rec.Synapse.synapse_type == 'in' else rec.PulseResponse.ex_qc_pass
                    spike_t = rec.StimPulse.first_spike_time

                    if not qc_pass:
                        print("qc fail: ", rec.PulseResponse.meta.get('qc_failures', 'no qc failures recorded'))
                        qc_pass_rec = False

                    if spike_t is None:
                        spike_t = rec.StimPulse.onset_time + 1e-3

                    # print(rec.data)
                    t0 = rec.PulseResponse.data_start_time - spike_t
                    ts = TSeries(data=rec.data, t0=t0, sample_rate=db.default_sample_rate)

                    t0 = rec.spike_data_start_time - spike_t
                    spike_ts = TSeries(data=rec.spike_data, t0=t0, sample_rate=db.default_sample_rate)

                    # arrange plots nicely
                    y0 = ts.time_slice(None, 0).median()
                    shift = (pulse_n * 35e-3 + (30e-3 if pulse_n > 8 else 0), -y0)
                    # plt.plot(ts.time_values + shift[0], ts.data + shift[1], linewidth=0.1)
                    # plt.axvline(x=shift[0])

                    index_before_spike = [list(ts.time_values).index(time) for time in ts.time_values if (time + shift[0]) >= (shift[0] - 0.002) and (time + shift[0]) < shift[0]]
                    index_after_spike = [list(ts.time_values).index(time) for time in ts.time_values if (time + shift[0]) <= (shift[0] + 0.01) and (time + shift[0]) > shift[0] + 0.002]

                    average_before_spike = np.nanmean([ts.data[i] + shift[1] for i in index_before_spike])
                    average_after_spike = np.nanmean([ts.data[i] + shift[1] for i in index_after_spike])
                    amp = average_after_spike - average_before_spike
                    responses.append(amp)

                    # if amp > 0.002:
                    #     amp_crossed_upper_bound = True

                if not qc_pass_rec or amp_crossed_upper_bound:
                    print({k + 1: responses[k] for k in list(pulses.keys())})
                    continue

                amps = {k + 1: responses[k] for k in list(pulses.keys())}
                normamps = np.array(list(amps.values()))

                # plt.show()

                pre_amps[pair.id][ind_freq, rec_delay].append(normamps)

                # create set of all cre_types for later printout
                unique_cre_types.add(pair.pre_cell.cre_type)
                unique_cre_types.add(pair.post_cell.cre_type)

                # get dynamics for pair
                PPR = pair.dynamics.paired_pulse_ratio_50hz
                try:
                    release_prob = pair.SynapseModel.ml_base_release_probability
                    results['Release_Prob_Est'] = release_prob
                except:
                    print("No Synapse model")

                results = results.append({'pair_id': str(pair),
                                          'pre_cell': pair.pre_cell.cre_type,  # previosuly .cell_class
                                          'post_cell': pair.post_cell.cre_type,
                                          'rec_id': recording.id,
                                          'clamp_mode': clamp_mode,
                                          'ind_freq': ind_freq,
                                          'rec_delay': rec_delay,
                                          'amps': np.array(list(amps.values())),
                                          'normamps': normamps,
                                          'Paired_Pulse_50hz': PPR,
                                          'stimuli': list(amps.keys()),
                                          'psc_amp': pair.synapse.psc_amplitude,
                                          'psp_amp': pair.synapse.psp_amplitude},
                                         ignore_index=True)


#JB: take the "results" dataframe and process it into a dict organized by synapse type
def organize_syn_dict(pulse_res_df):
    print(str(pulse_res_df))
    syn_dict = {}
    for index, row in results.iterrows():
        if row['rec_delay'] in [0.125, 0.126, 0.127, 0.128]:
            row['rec_delay'] = 0.125
        if row['rec_delay'] in [0.250,0.251, 0.252, 0.253]:
            row['rec_delay'] = 0.250
        syn_key = (row['pre_cell'], row['post_cell'])
        sim_key = (row['clamp_mode'], row['ind_freq'], row['rec_delay'])

        processed_amps = fill_pad(row['normamps'], row['stimuli'])

        if (syn_key in syn_dict):
            if row['pair_id'] in syn_dict[syn_key]: #need pair_IDs here
                if sim_key in syn_dict[syn_key][row['pair_id']]:
                    syn_dict[syn_key][row['pair_id']][sim_key] = np.vstack((syn_dict[syn_key][row['pair_id']][sim_key] , processed_amps))
                else:
                    syn_dict[syn_key][row['pair_id']][sim_key] = np.array([processed_amps])
            else:
                syn_dict[syn_key][row['pair_id']] = {sim_key: np.array([processed_amps])}
                syn_dict[syn_key][row['pair_id']]['Paired_Pulse_50Hz'] = row['Paired_Pulse_50hz']
                try:
                    syn_dict[syn_key][row['pair_id']]['Release_Prob_Est'] = row['Release_Prob_Est']
                except:
                    print("No Synapse model")
        else:
            syn_dict[syn_key] = {row['pair_id']: {sim_key: np.array([processed_amps])}}
            syn_dict[syn_key][row['pair_id']]['Paired_Pulse_50Hz'] = row['Paired_Pulse_50hz']
            syn_dict[syn_key]['pair_IDs'] = set()
        syn_dict[syn_key]['pair_IDs'].add(str(row['pair_id']))
    return syn_dict

# syn_key, pair_id, sim_key
def gen_measures(syn_dict):
    measures_dict = {}
    lower_threshold = -0.005
    for syn_key in syn_dict.keys():
        measures_dict[syn_key] = {}
        for pair_id in syn_dict[syn_key].keys():
            #skip entry listing pair_IDs
            no_spikes = True
            if pair_id == 'pair_IDs':
                continue
            print("gen_measures, pair: "+str(pair_id))
            measures_dict[syn_key][pair_id] = {'Paired_Pulse_50Hz': syn_dict[syn_key][pair_id]['Paired_Pulse_50Hz']}
            try:
                measures_dict[syn_key][pair_id]['Release_Prob_Est'] = syn_dict[syn_key][pair_id]['Release_Prob_Est']
            except:
                print("no")
                print("No Synapse model")
            first_spike_list = []

            #run loop once to get divisor for normalization
            for sim_key in syn_dict[syn_key][pair_id].keys():
                if (sim_key == 'Paired_Pulse_50Hz') or (sim_key =='Release_Prob_Est'):
                    continue
                try:
                    divisors = syn_dict[syn_key][pair_id][sim_key][:, 0]
                    print("printing divisors")
                    print(divisors)
                    for i in range(0, len(divisors)):
                        if divisors[i] < -1E-9:
                            if divisors[i] > lower_threshold:
                                first_spike_list.append(divisors[i])
                            else:
                                first_spike_list.append(lower_threshold)
                        else:
                            first_spike_list.append(-1E-9)
                except:
                    print("no entry")
            if len(first_spike_list) > 0:
                averaged_divisor = sum(first_spike_list)/len(first_spike_list)
                no_spikes = False
                print("nonzero spike count")

            first_fifth_ratios_50 = []
            first_second_ratios_50 = []
            firsts_50 = []
            fifths_50 = []
            seconds_50 = []
            ninths_50 = []
            areas_50 = []
            first_4s_50 = []
            last_4s_50 = []

            num_amps = 0
            num_failures = 0

            #run loop again for measures
            for sim_key in syn_dict[syn_key][pair_id].keys():
                if (sim_key == 'Paired_Pulse_50Hz') or (sim_key =='Release_Prob_Est'):
                    continue

                clamp, freq, delay = sim_key

                if freq == None or delay == None:
                    continue

                areas = []
                first_fifth_ratios = []
                first_second_ratios = []
                firsts = []
                seconds = []
                fifths = []
                ninths = []
                eighths = []
                first_4s = []
                last_4s = []
                normed_all = []
                for i in range(0, len(syn_dict[syn_key][pair_id][sim_key])):
                    normed_row = syn_dict[syn_key][pair_id][sim_key][i]

                    for i in range(0, len(normed_row)):

                        if normed_row[i] < -1E-9:
                            if i == 0:
                                num_amps += 1
                            if normed_row[i] < lower_threshold:
                                normed_row[i] = lower_threshold
                        else:
                            normed_row[i] = -1E-9
                            if i == 0:
                                num_failures += 1
                                num_amps += 1
                    print("Printing normed_row before")
                    print(normed_row)
                    normed_row = normed_row/averaged_divisor
                    normed_all.append(normed_row)

                    for i in range(0, 4):
                        first_4s.append(normed_row[i])
                    for i in range(8, 12):
                        last_4s.append(normed_row[i])

                    exceeding_bound = False
                    for response in normed_row:
                        if response > 7:
                            exceeding_bound = True

                    if exceeding_bound:
                        print(True)
                        continue

                    print("printing normed_row")
                    print(normed_row)
                    first = normed_row[0]
                    second = normed_row[1]
                    fifth = normed_row[4]
                    ninth = normed_row[8]
                    eighth = normed_row[9]
                    firsts.append(first)
                    seconds.append(second)
                    fifths.append(fifth)
                    eighths.append(eighth)
                    ninths.append(ninth)

                    if not math.isnan(first):
                        firsts.append(first)
                    if not math.isnan(second):
                        seconds.append(second)
                    if not math.isnan(fifth):
                        fifths.append(fifth)
                    if not math.isnan(fifth/first):
                        first_fifth_ratios.append(fifth/first)
                        print("first = "+str(first))
                        print("fifth = "+str(fifth))
                    if not math.isnan(second/first):
                        first_second_ratios.append(second/first)
                        print("first = "+str(first))
                        print("second = "+str(second))
                    if not math.isnan(np.sum(normed_row[0:8])):
                        areas.append(np.sum(normed_row[0:8]))

                if len(areas) > 0:

                    measures_dict[syn_key][pair_id][sim_key] = {'area_first_eight_mean': sum(areas)/len(areas)}
                    mean_first = sum(firsts)/len(firsts)
                    mean_second = sum(seconds)/len(seconds)
                    mean_fifth = sum(fifths)/len(fifths)
                    mean_ninth = sum(ninths)/len(ninths)
                    av_first_four = sum(first_4s)/len(first_4s)
                    av_last_four = sum(last_4s)/len(last_4s)
                    recovery = av_last_four - av_first_four
                    measures_dict[syn_key][pair_id][sim_key]['recovery'] = recovery
                    measures_dict[syn_key][pair_id][sim_key]['first_second_mean'] = mean_second/mean_first
                    measures_dict[syn_key][pair_id][sim_key]['first_fifth_mean'] = mean_fifth/mean_first
                    if len(eighths) > 0:
                        mean_eighth = sum(eighths)/len(eighths)
                        measures_dict[syn_key][pair_id][sim_key]['ninth_eight_mean'] = mean_ninth/mean_eighth
                    measures_dict[syn_key][pair_id][sim_key]['first_ninth_mean'] = mean_ninth/mean_first

                if abs(freq -50.0) < 1: #verify this works
                    print("50hz found")
                    if len(first_fifth_ratios) > 0:
                        first_fifth_ratios_50 = first_fifth_ratios_50 + first_fifth_ratios
                        print("First fifth 50Hz")
                    if len(first_second_ratios) > 0:
                        first_second_ratios_50 = first_second_ratios_50 + first_second_ratios
                        print("First fifth 50Hz")
                    if len(firsts) > 0:
                        firsts_50 = firsts_50 + firsts
                    if len(seconds) > 0:
                        seconds_50 = seconds_50 + seconds
                    if len(fifths) > 0:
                         fifths_50 = fifths_50 + fifths
                    if len(ninths) > 0:
                     ninths_50 = ninths_50 + ninths
                    if len(areas) > 0:
                        areas_50 = areas_50 + areas #is this right?
                        print("areas 50Hz")

                try:
                    mean_first_50 = sum(firsts_50) / len(firsts_50)
                    mean_second_50 = sum(seconds_50) / len(seconds_50)
                    mean_fifth_50 = sum(fifths_50) / len(fifths_50)
                    mean_ninth_50 = sum(ninths_50) / len(ninths_50)
                    first_fifth = mean_fifth_50 / mean_first_50
                    first_second = mean_second_50 / mean_first_50
                    first_ninth = mean_ninth_50 / mean_first_50
                    measures_dict[syn_key][pair_id]['release_prob_all'] = (num_amps - num_failures) / num_amps
                    measures_dict[syn_key][pair_id]['areas_50hz_mean'] = sum(areas_50) / len(areas_50)
                    measures_dict[syn_key][pair_id]['first_fifth_50hz_mean'] = first_fifth
                    measures_dict[syn_key][pair_id]['first_second_50hz_mean'] = first_second
                    measures_dict[syn_key][pair_id]['first_ninth_50hz_mean'] = first_ninth
                    print("added entries")
                except:
                    print("no entries")

    return measures_dict

#create array of 12 nan values and add in values from norm_amp_arr at
#locations of corresponding keys
def fill_pad(norm_amp_arr, arr_keys):
    arr_keys = [*range(1, 13)]
    arr = np.empty((12))*np.nan
    #arr[:] = np.nan
    num_allocated = 0
    for key in arr_keys:
        print("key is:")
        print(key)
        arr[key-1] = norm_amp_arr[num_allocated]
        num_allocated = num_allocated + 1 #move index in norm_amp_arr
    return arr


def process_amps(amp_arr, clamp_mode, vc_amp, ic_amp, normalize=True):
    if normalize:
        if (clamp_mode == 'ic'): #normalize current
            #out_arr = np.divide(amp_arr, ic_amp)
            if ic_amp == None:
                return (False, None)
            out_arr = pad_amps(amp_arr)
            print("ic_amp ="+str(ic_amp))
            #out_arr = out_arr -ic_amp
            out_arr = np.divide(out_arr, ic_amp)
            #out_arr = np.divide(out_arr, out_arr*ic_amp)
        elif(clamp_mode == 'vc'): #normalize voltage
            #out_arr = np.divide(amp_arr, vc_amp)
            if vc_amp == None:
                return (False, None)
            out_arr = pad_amps(amp_arr)
            #out_arr = out_arr -vc_amp #divide by the ic_amp
            out_arr = np.divide(out_arr, vc_amp)
        else:
            raise NameError('Unknown clamp_mode')
    else:
        out_arr = pad_amps(amp_arr)
    return (True, out_arr)


def pad_amps(amp_list):
    extracted_amplitudes = np.zeros(12)
    for i in range(12):
        if i in range(len(amp_list)):
            extracted_amplitudes[i] = amp_list[i]
        else:
            extracted_amplitudes[i] = np.nan
    return extracted_amplitudes


def calc_norm_amplitudes(results):

    initialAmp = {}
    # iterate over pairs
    for id in list(results.pair_id.unique()):
        subdat = results[results.pair_id==id]
        vcdat = subdat[subdat.clamp_mode == 'vc']
        icdat = subdat[subdat.clamp_mode == 'ic']
        initialAmp[id] = {'vc': np.array([vcdat.amps.iloc[i][0] for i in range(len(vcdat))]).mean(),
                          'ic': np.array([icdat.amps.iloc[i][0] for i in range(len(icdat))]).mean()}

    normAmp = []

    for i in range(len(results)):
        s = results.iloc[i]
        normAmp.append(np.array(s.amps) / initialAmp[s.pair_id][s.clamp_mode])

    results['normAmps'] = normAmp

    return results


def get_amps_first8(results, normalized=False):
    """
    Extracts response to first 8 pulses (no recovery pulses)
    """

    x = np.zeros((len(results), 8))

    for ix in range(len(results)):
        if normalized:
            amps = results.iloc[ix].normAmps
        else:
            amps = results.iloc[ix].amps

            if len(amps) >= 8:
                x[ix, :] = amps[:8]
            else:
                x[ix, :] = np.pad(amps, (0, 8-len(amps)), constant_values=np.nan)

    return x


test_dict = organize_syn_dict(results)
measures_dict = gen_measures(test_dict)
pickle.dump(measures_dict, open("./Data/Measures_1.3mM_in_Rodent.p", "wb"))

