
"""
Loads supervised classifier accuracy outputs, evaluates classifier performance and feature importance
across pre‑synaptic cell types, and generates boxplots and supervised weights ‑distribution histograms.

Author: jadepoir
"""

import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt

# accuracies = pickle.load(open("./Data/supervised_alg_in_rodent_phys.p", "rb")) # pre phys
# accuracies = pickle.load(open("./Data/supervised_alg_in_rodent_model_post.p", "rb")) # post model
accuracies = pickle.load(open('./Data/supervised_alg_in_rodent_model.p', 'rb')) # pre model

accuracies_dict = {}
accuracies_dict["baseline"] = []
alg_labels = ["gb", "lr", "adb", "mlp", "rf", "svm", "gmm"]
alg_dict = {}
for i, label in enumerate(alg_labels):
    accuracies_dict[label] = []
    alg_dict[i] = label

# for i, alg in enumerate(accuracies[1]):
#     accuracies_dict[alg] = accuracies[0][i][0]
#     for j in accuracies[0][i][1]:
#         print(j)
#         accuracies_dict["baseline"].append(j)

for alg in range(len(accuracies[0])):
    for i in range(len(accuracies[0][alg][0])):
        accuracies_dict[alg_dict[alg]].append(accuracies[0][alg][0][i])

for alg in accuracies_dict.keys():
    if alg != "baseline":
        mean_accuracy = np.mean(accuracies_dict[alg])
        sd_accuracy = np.std(accuracies_dict[alg])
        # print(f"{alg}: {mean_accuracy:.2f} +/- {sd_accuracy:.2f}")

for i in range(len(accuracies[0][0][1])):
    accuracies_dict["baseline"].append(accuracies[0][0][1][i])

# differences
significance = []
for alg in range(len(accuracies[0])):
    print(alg)
    count = 0
    for i in accuracies[0][alg][2]:
        if i <= 0:
            count += 1
    # print(accuracies[0][alg][2])
    significance.append(count / len(accuracies[0][alg][2]))

print(significance)

accuracies_dict_2 = copy(accuracies_dict)
del accuracies_dict_2["baseline"]
del accuracies_dict_2["gb"]
del accuracies_dict_2["svm"]
del accuracies_dict_2["rf"]
# del accuracies_dict_2["adb"]
del accuracies_dict_2["mlp"]
del accuracies_dict_2["gmm"]

# --- Type Accuracy (All Classifiers) ---

"""
fig, ax = plt.subplots()
plt.title("Pre-Type Accuracy")
ax.boxplot(accuracies_dict_2.values())
ax.set_xticklabels(accuracies_dict_2.keys())
ax.set_ylabel('Predictive Accuracy')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(bottom=0, top=0.1)
ax.set_ylim(bottom=0.2, top=1)
ax.axhline(y=accuracies_dict["baseline"][0], color='xkcd:dark blue', linestyle='--')
fig.set_size_inches((4.52, 3.456))
fig.set_dpi(1200)
fig.tight_layout()
# plt.savefig("./Pre-Type_Accuracy.svg", transparent=True)

plt.show()
"""

# --- Type Accuracy (Top 2 Classifiers) ---

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[5,1]}) # [1, 5] for post, [5, 1] for pre

# Plot the boxplots
flierprops = dict(marker='o', markersize=3, linestyle="none")
ax1.boxplot(accuracies_dict_2.values(), medianprops=dict(color="xkcd:teal blue"), flierprops=flierprops, widths=0.5)
ax2.boxplot(accuracies_dict_2.values(), medianprops=dict(color="xkcd:teal blue"), flierprops=flierprops, widths=0.5)
ax2.axhline(y=accuracies_dict["baseline"][0], color='xkcd:pumpkin orange', linestyle='--') # ***
fig.subplots_adjust(hspace=100)

# Set y-axis limits
# ax1.set_ylim(0.6, 0.7) # post
# ax2.set_ylim(0, 0.4) # post
ax1.set_ylim(0.3, 0.7) # pre
ax2.set_ylim(0, 0.2) # pre

# Hide spines
ax1.spines['bottom'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Adjust ticks
ax1.tick_params(bottom=False)
ax2.tick_params(top=False)

# Set consistent tick intervals
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))

# Add break markers
d = .5  # Proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=4, clip_on=False)
ax1.plot([0, 0], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [1, 1], transform=ax2.transAxes, **kwargs)

# Set x-ticks and labels only on the bottom plot
ax2.set_xticks(range(1, len(accuracies_dict_2.keys()) + 1))
ax2.set_xticklabels(accuracies_dict_2.keys())
ax1.tick_params(labelsize=6)
ax2.tick_params(labelsize=6)

# Set y-axis label on the bottom plot
# fig.text(0.04, 0.5, 'Predictive Accuracy', va='center', rotation='vertical')
# ax2.set_ylabel('Predictive Accuracy')

# Title
# fig.suptitle("Pre-Type Accuracy")

# Adjust figure size
fig.set_size_inches((1.285, 1.974))
fig.set_dpi(1200)
fig.tight_layout()

plt.savefig("./Figures/fig3/Pre-Type_Accuracy_Model_pre_lr_adb.svg", transparent=True)
plt.show()

features_dict = {
    0: "Baseline",
    1: "Mu_amp1",
    2: "Mu_amp2",
    3: "Mu_amp3",
    4: "Mu_amp4",
    5: "Sigma Baseline",
    6: "Sigma_amp",
    # 7: "SD"
}

features_dict_phys = {
    0: "areas",
    1: "release_prob",
    2: "first_fifth",
    3: "first_second",
    4: "recovery_50"
}

def normalize_coefficients(coeffs):
    total = np.sum(np.abs(coeffs))
    return (coeffs / total) * 100


# --- Supervised Weights Distributions

coeff_output = accuracies[2]

for i in range(len(features_dict_phys)):
    coef_0 = coeff_output[0][i] # pvalb
    coef_1 = coeff_output[1][i] # sst
    coef_2 = coeff_output[2][i] # vip
    # coef_3 = coeff_output[3][i]
    # coef_4 = coeff_output[4][i]
    # coef_5 = coeff_output[5][i]
    # coef_6 = coeff_output[6][i]

    fig, ax = plt.subplots()
    # plt.title(f"Coefficients for {features_dict[i]} Parameter")
    # Pre-types stuff
    ax.hist(coef_0, alpha=0.6, label='pvalb', color='xkcd:dark blue', bins="auto")
    ax.hist(coef_1, alpha=0.6, label='sst', color='xkcd:red orange', bins="auto")
    ax.hist(coef_2, alpha=0.6, label='vip', color='xkcd:golden yellow', bins="auto")

    # Post_types
    # ax.hist(coef_0, alpha=0.6, label='nr5a1', color='red', bins="auto")
    # ax.hist(coef_1, alpha=0.6, label='ntsr1', color='blue', bins="auto")
    # ax.hist(coef_2, alpha=0.6, label='pvalb', color='xkcd:dark blue', bins="auto")
    # ax.hist(coef_3, alpha=0.6, label='sim1', color='green', bins="auto")
    # ax.hist(coef_4, alpha=0.6, label='sst', color='xkcd:red orange', bins="auto")
    # ax.hist(coef_5, alpha=0.6, label='tlx3', color='grey', bins="auto")
    # ax.hist(coef_6, alpha=0.6, label='vip', color='xkcd:golden yellow', bins="auto")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=6)
    # fig.suptitle(features_dict[i])

    fig.set_dpi(1200)
    fig.set_size_inches((1.324, 0.955))  # in paper
    fig.tight_layout()
    plt.savefig(f"./Figures/fig3/Coefficients_for_{features_dict[i]}_Parameter.svg", transparent=True)
    # plt.savefig(f"./Figures/sup1/Coefficients_for_{features_dict_phys[i]}_Parameter.svg", transparent=True) # phys params

    # plt.show()