
"""
Trains and evaluates multiple supervised classifiers to predict inhibitory cell type from
SRP model parameters using a stratified, k-fold bootstrap.

Authors: jgben, jadepoir
"""

import os
import pickle
import random
import statistics
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from collections import Counter
from multiprocessing import Pool
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier as RF,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)

# ----------------------------------------------------------------------------------

def iter_function(inputs):
    max_iter, categories_1, pca_arr_1, fits_dict = inputs
    counter_dict = Counter(categories_1)

    inverted_dict = {value:key for key,value in Counter(categories_1).items()}
    largest_class = inverted_dict[max(inverted_dict.keys())]
    size_largest_class = max(inverted_dict.keys())

    other_class_dict = {other_class - 1:size_largest_class - counter_dict[other_class] for other_class in counter_dict.keys() if other_class != 0}

    print("------------------------")

    num_params = len(pca_arr_1[0])
    alg_labels = ["gb", "lr", "adb", "mlp", "rf", "svm", "gmm"]
    alg_outputs = [[-1 for i in range(3)] for j in range(0, len(alg_labels))]
    coef_outputs = [[[] for j in range(num_params)] for i in range(len(set(categories_1)))]
    perfo_outputs = [[-1 for i in range(len(set(categories_1)))] for j in range(0, len(alg_labels))]

    prob_dict = {index: [] for index in range(len(pca_arr_1))}

    for i in range(0, max_iter):

        pca_arr = pca_arr_1
        categories = categories_1

        # print(f"Length categories: {len(categories)}")
        cross_accuracies = [[] for i in range(0, len(alg_labels))]
        cross_baselines = [[] for i in range(0, len(alg_labels))]
        cross_perfo = [[] for i in range(0, len(alg_labels))]

        # split data into stratified folds
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        print("Shuffle " + str(i))

        for i, (train_index, test_index) in enumerate(skf.split(pca_arr, categories)):

            # make train and test sets for this fold
            train_data_1, train_targets_1 = pca_arr[train_index], categories[train_index]
            test_data_1, test_targets_1 = pca_arr[test_index], categories[test_index]

            # Balance test data to vip
            num_pre_test = Counter(test_targets_1)

            pv_positions = random.sample(range(num_pre_test[0]), num_pre_test[2])
            sst_positions = random.sample(range(num_pre_test[0], num_pre_test[0] + num_pre_test[1]), num_pre_test[2])
            vip_positions = list(range(num_pre_test[0] + num_pre_test[1], num_pre_test[0] + num_pre_test[1] + num_pre_test[2]))

            test_index_final = np.array([test_index[array_ind] for array_ind in pv_positions + sst_positions + vip_positions])

            test_targets = [0] * num_pre_test[2] + [1] * num_pre_test[2] + [2] * num_pre_test[2]
            test_targets = np.array(test_targets)

            test_data = np.array([test_data_1[array_ind] for array_ind in pv_positions + sst_positions + vip_positions])
            # test_data = np.array([test_data_1[array_ind] for array_ind in zero_pos + one_pos + two_pos + three_pos + four_pos + five_pos + six_pos])

            num_pre_types = Counter(train_targets_1)

            sst_positions = range(num_pre_types[0], num_pre_types[0] + num_pre_types[1])
            vip_positions = range(num_pre_types[0] + num_pre_types[1],
                                  num_pre_types[0] + num_pre_types[1] + num_pre_types[2])

            train_targets = [0] * num_pre_types[0] + [1] * num_pre_types[0] + [2] * num_pre_types[0]
            train_targets = np.array(train_targets)

            train_data = np.array([np.array(arr) for arr in train_data_1[:num_pre_types[0]]])

            sst_indices = random.choices(sst_positions, k=num_pre_types[0])
            vip_indices = random.choices(vip_positions, k=num_pre_types[0])

            train_data_sst = train_data_1[sst_indices, :]
            train_data_vip = train_data_1[vip_indices, :]

            train_data = np.vstack([train_data, train_data_sst])
            train_data = np.vstack([train_data, train_data_vip])

            clf_lr = LogisticRegression(max_iter=20000)
            clf_svm = SVC()
            clf_gb = GradientBoostingClassifier(n_estimators=500)
            clf_rf = RF()
            clf_ad = AdaBoostClassifier()
            clf_mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=500)
            clf_gmm = GaussianNB()

            clf_list = [clf_gb, clf_lr, clf_ad, clf_mlp, clf_rf, clf_svm, clf_gmm]
            for index in range(0, len(clf_list)):
                clf = clf_list[index]
                # multi_target_svc = MultiOutputClassifier(clf, n_jobs=-1)
                clf.fit(train_data, train_targets)

                if clf == clf_lr:
                    # Coefficients and Odds Ratios
                    coefficients = clf.coef_
                    odds_ratios = np.exp(coefficients)

                    # Display feature importance using coefficients and odds ratios
                    feature_importance_0 = pd.DataFrame({
                        'Feature': [*range(num_params)],
                        'Coefficient': coefficients[0],
                        'Odds Ratio': odds_ratios[0]
                    })
                    feature_importance_1 = pd.DataFrame({
                        'Feature': [*range(num_params)],
                        'Coefficient': coefficients[1],
                        'Odds Ratio': odds_ratios[1]
                    })
                    feature_importance_2 = pd.DataFrame({
                        'Feature': [*range(num_params)],
                        'Coefficient': coefficients[2],
                        'Odds Ratio': odds_ratios[2]
                    })

                if clf == clf_gmm:
                    predict_prob = clf.predict_proba(test_data)
                    zipped_prob = list(zip(test_index_final, predict_prob))
                    for zipped in zipped_prob:
                        prob_dict[zipped[0]].append(zipped[1])

                accuracy = clf.score(test_data, test_targets)
                print(accuracy)

                pred = clf.predict(test_data)

                cm = confusion_matrix(test_targets, pred)

                fp = cm.sum(axis=0) - np.diag(cm)  # False Positives for each class
                fn = cm.sum(axis=1) - np.diag(cm) # False Negatives for each class
                tp = np.diag(cm)
                tn = [np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i,i] for i in range(cm.shape[0])]

                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                fpr = fp / (fp + tn)
                fnr = fn / (fn + tp)

                cross_perfo[index].append([sensitivity, specificity, fpr, fnr])

                # test baseline accuracy on this partition
                dummy_clf = DummyClassifier(strategy='most_frequent')
                dummy_clf.fit(train_data, train_targets)  # model like comparison
                # dummy_clf.fit(test_data, test_targets) #how well does baseline do if it truly knows the test data
                baseline_accuracy = dummy_clf.score(test_data, test_targets)  # to test against knowing test set
                predictions = dummy_clf.predict(test_data)

                # print(index)
                # alg_outputs[index][0].append(accuracy)
                # alg_outputs[index][1].append(baseline_accuracy)
                cross_accuracies[index].append(accuracy)
                # test_accuracies.append(accuracy)

                # print("printing baseline predictions")
                # print(predictions)
                # baseline_accuracies.append(baseline_accuracy)
                cross_baselines[index].append(baseline_accuracy)
                # print("printing baseline_accuracy")
                # print(baseline_accuracy)
                # differences.append(accuracy-baseline_accuracy)

        for i in range(0, len(alg_labels)):
            accuracy = statistics.mean(cross_accuracies[i])
            baseline = statistics.mean(cross_baselines[i])
            difference = accuracy - baseline
            performance = np.nanmean(cross_perfo[i], axis=0)
            # print(accuracy)

            if alg_outputs[index][0] == -1:  # if not filled
                alg_outputs[i][0] = [accuracy]
                alg_outputs[i][1] = [baseline]
                alg_outputs[i][2] = [difference]
                perfo_outputs[i][0] = [[perfo_element[0] for perfo_element in performance]]
                perfo_outputs[i][1] = [[perfo_element[1] for perfo_element in performance]]
                perfo_outputs[i][2] = [[perfo_element[2] for perfo_element in performance]]

                for j in range(num_params):
                    coef_outputs[0][j] = [feature_importance_0["Coefficient"][j]]
                    coef_outputs[1][j] = [feature_importance_1["Coefficient"][j]]
                    coef_outputs[2][j] = [feature_importance_2["Coefficient"][j]]

            else:
                # print(alg_outputs[i][0])
                alg_outputs[i][0].append(accuracy)
                alg_outputs[i][1].append(baseline)
                alg_outputs[i][2].append(difference)
                perfo_outputs[i][0].append([perfo_element[0] for perfo_element in performance])
                perfo_outputs[i][1].append([perfo_element[1] for perfo_element in performance])
                perfo_outputs[i][2].append([perfo_element[2] for perfo_element in performance])
                for j in range(num_params):
                    coef_outputs[0][j].append(feature_importance_0["Coefficient"][j])
                    coef_outputs[1][j].append(feature_importance_1["Coefficient"][j])
                    coef_outputs[2][j].append(feature_importance_2["Coefficient"][j])

    return (alg_outputs, alg_labels, coef_outputs, perfo_outputs, prob_dict)

def accuracy_bootstrap_stratified(params, row_labels, fits_ids, partition_size=10):
    raw_data = params.copy()

    scaler = StandardScaler()
    scaler.fit(raw_data)
    scaled_arr = scaler.transform(raw_data)
    pca_arr_1 = scaled_arr

    fits_dict = dict(zip([str(arr) for arr in pca_arr_1], fits_ids))
    enc = LabelEncoder()
    categories_1 = np.asarray([row for row in row_labels])

    one_hot_targets = enc.fit_transform(categories_1)
    categories_1 = one_hot_targets

    if __name__ == '__main__':
        max_iteration = 525
        num_processes = 10 # Number of parallel processes
        num_params = len(pca_arr_1[0])

        inputs = [[int(max_iteration / num_processes), categories_1, pca_arr_1, fits_dict]] * num_processes

        with Pool(processes=num_processes) as pool:
            results = pool.map(iter_function, inputs)

        print(f"Total iterations completed: {max_iteration / num_processes}")

        pickle_path = './Data/supervised_alg_in_rodent_model.p'

        if os.path.exists(pickle_path):
            alg_outputs, alg_labels, coef_outputs, perfo_outputs, final_prob_dict = pickle.load(open(pickle_path, 'rb'))
        else:
            alg_labels = ['gb', 'lr', 'adb', 'mlp', 'rf', 'svm', "gmm"]
            alg_outputs = [[[] for j in range(3)] for j in range(len(alg_labels))]
            coef_outputs = [[[] for j in range(num_params)] for i in range(len(set(categories_1)))]
            perfo_outputs = [[[] for j in range(len(set(categories_1)))] for i in range(len(alg_labels))]
            final_prob_dict = {index: [] for index in range(len(pca_arr_1))}

        for num_pro in range(len(results)): # num_processes
            for iter in range(len(results[num_pro])): # alg_outputs, alg_labels, coef, perfo, prob_dict
                if iter == 0:
                    for alg in range(len(results[num_pro][iter])): # alg
                        for i in range(int(max_iteration / num_processes)):
                            alg_outputs[alg][0].append(results[num_pro][iter][alg][0][i])
                            alg_outputs[alg][1].append(results[num_pro][iter][alg][1][i])
                            alg_outputs[alg][2].append(results[num_pro][iter][alg][2][i])
                if iter == 2:
                    for type in range(len(results[num_pro][iter])):
                        for i in range(int(max_iteration / num_processes)):
                            for j in range(num_params):
                                coef_outputs[type][j].append(results[num_pro][iter][type][j][i])
                if iter == 3:
                    for alg in range(len(results[num_pro][iter])): # alg
                        for i in range(int(max_iteration / num_processes)):
                            perfo_outputs[alg][0].append(results[num_pro][iter][alg][0][i])
                            perfo_outputs[alg][1].append(results[num_pro][iter][alg][1][i])
                            perfo_outputs[alg][2].append(results[num_pro][iter][alg][2][i])
                if iter == 4:
                    for i in range(int(max_iteration / num_processes)):
                        for j in results[num_pro][iter].keys():
                            if len(results[num_pro][iter][j]) == 1:
                                final_prob_dict[j].append(results[num_pro][iter][j][0])
                            elif len(results[num_pro][iter][j]) > 1:
                                for prob in results[num_pro][iter][j]:
                                    final_prob_dict[j].append(prob)

        # print(f"Final prob dict: {final_prob_dict}")
        with open(
                './Data/supervised_alg_in_rodent_model.p', 'wb') as handle:
            pickle.dump((alg_outputs, alg_labels, coef_outputs, perfo_outputs, final_prob_dict), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------------------------------------

fits_pre_types = []
srp_fits = []
identities = []
fits_ids = []

directory = "./Data/srp_fits_in_rodent/"

for i in range(1, len(os.listdir(directory))):
    file_name = os.listdir(directory)[i]
    cell_type = file_name.split('_')[0]
    cell_id = file_name.split('_')[3:]
    cell_id = f"{cell_id[0]}_{cell_id[1]}_{cell_id[2].split('.p')[0]}"
    key = (cell_type, cell_id)
    pickle_file = open(directory + file_name, "rb")
    params1 = pickle.load(pickle_file)
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus = params1
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

    fitted_params = [mu_baseline, *mu_amps, sigma_baseline, sigma_amps]

    if sigma_baseline == 6 and sigma_amps == 1000:
        continue

    if sigma_baseline == None:
        continue

    fits_ids.append(cell_id)
    srp_fits.append(fitted_params)
    identities.append(key[0])

row_labels = np.array(identities)
fits_array = np.array(srp_fits)

accuracy_bootstrap_stratified(fits_array, row_labels, fits_ids)