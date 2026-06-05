
"""
Extracts 12‑pulse synaptic response amplitudes from mouse inhibitory synapses in the aisynphys database,
applies quality control, organizes the normalized responses by synapse type, and saves the resulting
short‑term plasticity data structure to a pickle file.

Authors:
    jgben — original implementation
    jadepoir — modifications for inhibitory project (2026)
"""

import pandas
import pickle
from aisynphys.dynamics import *
from neuroanalysis.data import TSeries
from aisynphys.database import SynphysDatabase

# Load database
db = SynphysDatabase.load_sqlite("D:/datasets/database/synphys_r2.1_full.sqlite", readonly=True)

# Load all synapses associated with mouse V1 projects
pairs = db.pair_query(
    synapse=True,
    species='mouse',
    synapse_type='in',
    #acsf='2mM Ca & Mg',  # '1.3mM Ca & 1mM Mg' or '2mM Ca & Mg'
    acsf='1.3mM Ca & 1mM Mg',
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
#create set of cre types for later print out
unique_cre_types = set()

for ix in range(len(pairs)):
    print('processing pair {} of {}'.format(ix + 1, len(pairs)))
    pair = pairs[ix]

    q = pulse_response_query(pair, db=db, data=True, spike_data=True)
    sorted_recs = sorted_pulse_responses(q.all())

    if sorted_recs != {}:  # if any recordings have a fit

        # Loop through all recordings
        for key, recs in sorted_recs.items():
            clamp_mode, ind_freq, rec_delay = key

            for recording, pulses in recs.items():
                if 1 not in pulses or 2 not in pulses:
                    continue

                if len(pulses.keys()) != 12:
                    continue

                responses = []
                qc_pass_rec = True
                amp_crossed_upper_bound = False

                if clamp_mode == 'vc':
                    amps = {k + 1: r.PulseResponseFit.fit_amp for k, r in pulses.items()}
                    if not isinstance(pair.synapse.psc_amplitude, float):
                        continue
                    first_val = list(amps.values())[0]
                    normamps = np.array(list(amps.values()))

                elif clamp_mode == 'ic':
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

                        t0 = rec.PulseResponse.data_start_time - spike_t
                        ts = TSeries(data=rec.data, t0=t0, sample_rate=db.default_sample_rate)

                        t0 = rec.spike_data_start_time - spike_t
                        spike_ts = TSeries(data=rec.spike_data, t0=t0, sample_rate=db.default_sample_rate)

                        # arrange plots nicely
                        y0 = ts.time_slice(None, 0).median()
                        shift = (pulse_n * 35e-3 + (30e-3 if pulse_n > 8 else 0), -y0)

                        index_before_spike = [list(ts.time_values).index(time) for time in ts.time_values if
                                              (time + shift[0]) >= (shift[0] - 0.002) and (time + shift[0]) < shift[0]]
                        index_after_spike = [list(ts.time_values).index(time) for time in ts.time_values if
                                             (time + shift[0]) <= (shift[0] + 0.01) and (time + shift[0]) > shift[0] + 0.002]

                        average_before_spike = np.nanmean([ts.data[i] + shift[1] for i in index_before_spike])
                        average_after_spike = np.nanmean([ts.data[i] + shift[1] for i in index_after_spike])
                        amp = average_after_spike - average_before_spike
                        responses.append(amp)

                        if amp > 0.002:
                            amp_crossed_upper_bound = True

                    if not qc_pass_rec or amp_crossed_upper_bound:
                        print({k + 1: responses[k] for k in list(pulses.keys())})
                        continue

                    amps = {k + 1: responses[k] for k in list(pulses.keys())}
                    normamps = np.array(list(amps.values()))

                #create set of all cre_types for later printout
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
                                          'pre_cell': pair.pre_cell.cre_type, #previosuly .cell_class
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
        #bin rec_delay values, note this edits the input dataframe so it should only be run once per
        #use of the script, fix this for future
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
                    print("on row number: "+str(index))
                    print(processed_amps)
                    print(syn_dict[syn_key][row['pair_id']][sim_key])
                    syn_dict[syn_key][row['pair_id']][sim_key] = np.vstack((syn_dict[syn_key][row['pair_id']][sim_key] , processed_amps))
                else:
                    syn_dict[syn_key][row['pair_id']][sim_key] = np.array([processed_amps])
            else:
                syn_dict[syn_key][row['pair_id']]= {sim_key: np.array([processed_amps])}
        else:
            syn_dict[syn_key] = {row['pair_id']: {sim_key: np.array([processed_amps])}}
            syn_dict[syn_key]['pair_IDs'] = set()
        syn_dict[syn_key]['pair_IDs'].add(str(row['pair_id']))
    return syn_dict


#create array of 12 nan values and add in values from norm_amp_arr at
#locations of corresponding keys
def fill_pad(norm_amp_arr, arr_keys):
    arr = np.empty((12))*np.nan
    num_allocated = 0
    for key in arr_keys:
        print("key is:")
        print(key)
        arr[key-1] = norm_amp_arr[num_allocated]
        num_allocated = num_allocated + 1
    return arr


def process_amps(amp_arr, clamp_mode, vc_amp, ic_amp, normalize=True):
    if normalize:
        if (clamp_mode == 'ic'): #normalize current
            if ic_amp == None:
                return (False, None)
            out_arr = pad_amps(amp_arr)
            print("ic_amp ="+str(ic_amp))
            out_arr = np.divide(out_arr, ic_amp)
        elif(clamp_mode == 'vc'): #normalize voltage
            if vc_amp == None:
                return (False, None)
            out_arr = pad_amps(amp_arr)
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


test_dict = organize_syn_dict(results)
pickle.dump(test_dict, open("Extracted_STP_1.3mM_Rodent.p", "wb"))