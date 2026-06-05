
"""
Fits Spike Response Plasticity (SRP) models to normalized response data for each inhibitory
mouse connection, using global optimization across multiple initial conditions, and saves the
best‑fitting SRP parameters for every synaptic pair.

Authors:
    jgben — original implementation
    jadepoir — modifications for inhibitory project (2026)
"""

import math
import pickle
import multiprocessing
from scipy.optimize import shgo
from srplasticity.inference import *

# Number of base functions summed to construct the efficacy kernels
num_bf = 4

def _default_parameter_bounds(mu_taus, sigma_taus):
    """ returns default parameter boundaries for the SRP fitting procedure """
    if num_bf == 4:
        return [
            (-6, 6),  # mu baseline
            *[(-50, 50), (-150, 150), (-1000, 1001), (-3000, 3000)],  # hand specify mu kernel bounds
            (-6, 6),  # sigma baseline
            *[(-1000, 1001)],  # sigma amps
            (0.001, 100),  # sigma scale
        ]


def _convert_fitting_params(x, mu_taus, sigma_taus, mu_scale=None):
    """
    Converts a vector of parameters for fitting `x` and independent variables
    (time constants and mu scale) to a vector that can be passed an an input
    argument to `ExpSRP` class
    """

    # Check length of time constants
    nr_mu_exps = len(mu_taus)
    nr_sigma_exps = len(sigma_taus)

    # Unroll list of initial parameters
    mu_baseline = x[0]
    mu_amps = x[1: 1 + nr_mu_exps]
    sigma_baseline = x[1 + nr_mu_exps]
    sigma_amps = x[2 + nr_mu_exps: 2 + nr_mu_exps + nr_sigma_exps]
    sigma_scale = x[-1]

    return (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )


def mse_loss_v3(target_vals, mean_predicted):

    """
    Stand in Mean Squared error for training loss in first phase
    :param target_vals: (np.array) set of amplitudes
    :param mean_predicted: (np.array) set of means
    """

    loss = []

    for key in target_vals.keys():
        if type(target_vals[key][0]) == np.float64:
            run_arr = target_vals[key]
            run_err = []

            if not np.isscalar(run_arr):
                for j in range(0, len(run_arr)):
                    run_err.append(math.pow((run_arr[j] - mean_predicted[key][j]), 2))
                loss.append(run_err)
        else:
            for i in range(0, len(target_vals[key])):
                run_arr = target_vals[key][i]  # get amplitudes from a single run
                run_err = []

                if not np.isscalar(run_arr):
                    for j in range(0, len(run_arr)):
                        run_err.append(math.pow((run_arr[j] - mean_predicted[key][j]), 2))
                    loss.append(run_err)

    total_mse_loss = np.nanmean(loss)

    return total_mse_loss


def _objective_function(x, *args, phase=0):
    """
    Objective function for scipy.optimize.minimize
    :param x: parameters for SRP model as a list or array:
                [mu_baseline, *mu_amps,
                sigma_baseline, *sigma_amps, sigma_scale]
    :param phase: 0 indicates fitting only mu amps and mu baseline for fixed
                    sigmas, 1 indicates fitting sigma params for fixed mu params
    :param args: target dictionary and stimulus dictionary
    :return: total loss to be minimized
    """
    # Unroll arguments
    target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fixed_baseline, fixed_amps, phase = args

    # Initialize model
    if phase == 0:  # fit mu params with fixed sigma params

        exp_amps = x[1:]
        new_x = np.append(x, [fixed_baseline])  # add fixed sigma params
        new_x = np.append(new_x, fixed_amps)
        model = ExpSRP(*_convert_fitting_params(new_x, mu_taus, sigma_taus))

    elif phase == 1:  # fit sigma params with fixed mu params
        # new_x = fixed_baseline + fixed_amps + x
        new_x = np.append([fixed_baseline], fixed_amps)  # add fixed sigma params
        new_x = np.append(new_x, x)
        model = ExpSRP(*_convert_fitting_params(new_x, mu_taus, sigma_taus))
    else:
        print("Error, undefined phase (ex: fixed mu params, fixed sigma params)")
    # compute estimates
    mean_dict = {}
    sigma_dict = {}
    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], sigma_dict[key], _ = model.run_ISIvec(ISIvec)

    # return loss
    if loss == "default":
        if phase == 0:
            return mse_loss_v3(target_dict, mean_dict)
            # return _total_loss(target_dict, mean_dict, sigma_dict)
        else:
            return mse_loss_v3(target_dict, mean_dict)
            # return _total_loss(target_dict, mean_dict, sigma_dict)

    else:
        raise ValueError(
            "Invalid loss function. Check the documentation for valid loss values"
        )


def fit_srp_model(
        stimulus_dict,
        target_dict,
        mu_taus,
        sigma_taus,
        initial_mu_baseline=[0],
        initial_mu=[0.01] * num_bf,
        initial_sigma_baseline=[-1.8],
        initial_sigma=[0.1] * num_bf,
        sigma_scale=[4],
        mu_scale=None,
        bounds="default",
        loss="default",
        algo="BFGS",
        **kwargs
):
    # default: algo="L-BFGS-B" but fix bounds
    """
    Fitting the SRP model to data using scipy.optimize.minimize
    :param initial_guess: list of parameters:
            [mu_baseline, *mu_amps,sigma_baseline, *sigma_amps, sigma_scale]
    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :param target_dict: mapping of protocol keys to response matrices
    :param mu_taus: predefined time constants for mean kernel
    :param sigma_taus: predefined time constants for sigma kernel
    :param mu_scale: mean scale, defaults to None for normalized data
    :param bounds: bounds for parameters
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param algo: Algorithm for fitting procedure
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.minimize
    """

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)

    optimizer_res = shgo(
        _objective_function,
        bounds=bounds[0:9],
        args=(
        target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma, 0),
        iters=1,
        **kwargs
    )

    params = _convert_fitting_params(list(optimizer_res.x), mu_taus,
                                     sigma_taus)

    fitted_mu_baseline = params[0]
    fitted_mu_amps = params[1]
    fitted_sigma_baseline = params[3]
    fitted_sigma_amp = params[4]

    output = (fitted_mu_baseline, fitted_mu_amps, mu_taus, fitted_sigma_baseline, fitted_sigma_amp, sigma_taus)

    return output, optimizer_res

# ----------------------------------------------------------------------------------------------------------------------

max_threshold = -0.005

measures_name = "Extracted_STP_1.3mM_in_Rodent.p" # Extracted_STP_1.3mM_in_Human.p
pickle_file = open(measures_name, "rb")
recordings = pickle.load(pickle_file)

for type_pair in recordings.keys():
    type1, type2 = type_pair

    chosen_dict = recordings[type_pair]

    target_dict = {}
    training_stim_dict = {}

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
                            divisor = chosen_dict[pair_id][protocol][i, 0]
                            if divisor < -1E-9:
                                if divisor > max_threshold:
                                    first_spike_list.append(divisor)
                            else:
                                first_spike_list.append(-1E-9)

            if len(first_spike_list) > 0:
                # apply normalisation
                averaged_divisor = sum(first_spike_list) / len(first_spike_list)
            else:
                averaged_divisor = first_spike_list[0]
                pass
            for protocol in chosen_dict[pair_id]:
                if not isinstance(protocol, int):
                    clamp, freq, delay = protocol

                    if freq == None or delay == None:
                        continue

                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            added_row = chosen_dict[pair_id][protocol][i, :]
                            valid_row = True
                            for n in range(0, len(added_row)):
                                if chosen_dict[pair_id][protocol][i, n] > -1E-9:
                                    added_row[n] = -1E-9
                                if chosen_dict[pair_id][protocol][i, n] < max_threshold:
                                    valid_row = False
                            if not valid_row:
                                continue
                            normed_row = added_row / averaged_divisor

                            exceeding_norm = False
                            for response in normed_row:
                                if response > 7:
                                    exceeding_norm = True

                            if exceeding_norm == True:
                                continue

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
                                        delay * 1000] + [1000 / freq] * 3  # should be [0] + ...
                                except:
                                    training_stim_dict[pair_id] = {
                                        (freq, delay): [0] + [1000 / freq] * 7 + [delay * 1000] + [1000 / freq] * 3}
    if num_bf == 4:
        mu_kernel_taus = [5, 15, 200, 4000]
        sigma_kernel_taus = [400]

    fitted_params = {}

    def fitting_function(arg_list):
        pair_ids, target_dict, training_stim_dict = arg_list

        for pair_id in pair_ids:
            for type_pair in recordings.keys():
                if pair_id in recordings[type_pair]:
                    type1, type2 = type_pair
            target_clean = {}
            for key in target_dict[pair_id]:
                if key != 'pair_ID':
                    target_clean[key] = target_dict[pair_id][key]
            stim_dict = training_stim_dict[pair_id]

            best_loss = None
            best_vals = None
            print(f"###### {pair_id} ######")
            for i in range(-6, 6):
                print(i)
                bounds = _default_parameter_bounds(mu_kernel_taus, sigma_kernel_taus)
                bounds[0] = (i, i+1)
                bounds[5] = (i, i+1)
                srp_params, optimizer_res = fit_srp_model(stim_dict, target_clean, mu_kernel_taus, sigma_kernel_taus, bounds=bounds)
                print(srp_params)

                if best_loss == None:
                    best_loss = optimizer_res.fun
                    best_vals = srp_params
                elif best_loss > optimizer_res.fun:
                    best_loss = optimizer_res.fun
                    best_vals = srp_params

            print(f"Best loss: {best_loss}")
            print(f"Best vals: {best_vals}")
            folder_name = f"srp_fits_in_rodent"
            new_pair_id = pair_id.replace(" ", "_")
            new_pair_id = new_pair_id.replace("<", "")
            new_pair_id = new_pair_id.replace(">", "")
            print(new_pair_id)
            name = folder_name + "/" + str(type1) + "_" + str(type2) + "_" + str(new_pair_id) + ".p"
            pickle.dump(best_vals, open(name, "wb"))


    def list_split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    if __name__ == '__main__':
        pair_id_list = [pair_id for pair_id in list(target_dict.keys())]

        arg_list = list(list_split(pair_id_list, 4))
        arg_list = [(pair_ids, target_dict, training_stim_dict) for pair_ids in arg_list]

        pool = multiprocessing.Pool()

        # Map the function over the arguments and get the results
        pool.map(fitting_function, arg_list)

        # Close the pool
        pool.close()
        pool.join()