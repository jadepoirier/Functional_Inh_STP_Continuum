
"""
Visualizes the distribution of silhouette scores from real versus artificial clustering runs
to assess the significance of observed cluster structure.

Author: jadepoir
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# scs = pickle.load(open("./Data/sc_ex_rodent_2clus_3000.p", "rb"))
scs = pickle.load(open("./Data/sc_in_rodent_2clus_3000.p", "rb"))
# scs = pickle.load(open("./Data/sc_in_human_2clus_3000.p", "rb"))

true_sc = [sc[0] for sc in scs if len(sc) == 4]
art_sc = [sc[2] for sc in scs if len(sc) == 4]

true_sc.append(scs[0][0])
art_sc.append(scs[0][1])

difference_list = [x - y for x, y in zip(true_sc, art_sc)]
p = Counter(np.array(difference_list) < 0)[True]/len(difference_list)

fig, ax = plt.subplots()

ax.hist(art_sc, bins=100, color="#1f77b4", alpha=0.6)
ax.tick_params(labelsize=6)

ax.spines[['right', 'top']].set_visible(False)
ax.vlines(x=np.nanmean(true_sc), ymin=0, ymax=200, color="#55c666", alpha=0.6)

fig.set_dpi(1200)
# fig.set_size_inches(0.67, 0.602)

plt.savefig("./Figures/fig2/SC_distribution_in_rodent.svg", transparent=True)
# plt.show()