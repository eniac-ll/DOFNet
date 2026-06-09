import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


font_size = 19
label_size = 16
bar_width = 0.4

colors_3 = ['#5ed8f8', '#a9aeff', '#9ff0e3']
colors_4 = ['#4e649d', '#c0ecae', '#e7bdc7', '#b8a8cf']

data1 = [60, 41, 30]
labels1=['Children', 'Adults', 'Elderly']

data2 = [100, 110, 120, 120]
labels2 = ['Normal', 'Mild', 'Moderate', 'Severe']

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 5),
    gridspec_kw={'width_ratios': [1, 1]}
)

wedges1, texts1, autotexts1 = axes[0].pie(
    data1,
    labels=None,
    colors=colors_3,
    autopct='%1.2f%%',
    textprops={'fontsize': label_size},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}  # ✅ 分割线
)

axes[0].set_aspect('equal')

axes[0].legend(
    wedges1,
    labels1,
    loc="center left",
    bbox_to_anchor=(0.92, 0.5),
    fontsize=label_size
)

x = np.arange(len(labels2))

bars = axes[1].bar(
    x,
    data2,
    color=colors_4,
    width=bar_width
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(labels2, fontsize=15)
axes[1].tick_params(axis='y', labelsize=label_size)

for bar in bars:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height}',
        ha='center',
        va='bottom',
        fontsize=label_size
    )

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

axes[0].text(
    0.6, -0.22, "(a) Age Distribution",
    transform=axes[0].transAxes,
    ha='center',
    va='center',
    fontsize=20
)
axes[1].text(
    0.5, -0.22, "(b) Disease Severity Distribution",
    transform=axes[1].transAxes,
    ha='center',
    va='center',
    fontsize=20
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figs/pie.png")
plt.show()