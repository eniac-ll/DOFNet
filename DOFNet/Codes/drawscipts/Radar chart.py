import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# =========================
file_paths = [
    r"otherconv.xlsx",
    r"Overview.xlsx",
    r"Focus.xlsx",
    r"numofgroups.xlsx"
]

subplot_titles = ["(a)", "(b)", "(c)", "(d)"]

metrics = ["FLOPs", "ACC", "SE", "F1", "Kappa"]
lower_better_metrics = ["FLOPs"]
inner = 0.62

# =========================
def load_and_normalize_excel(file_path, metrics, lower_better_metrics, inner=0.62):
    df = pd.read_excel(file_path)
    df.columns = df.columns.astype(str).str.strip()

    model_col = df.columns[0]
    models = df[model_col].astype(str).tolist()

    data = df[metrics].apply(pd.to_numeric, errors='coerce')

    norm_data = data.copy()
    for col in metrics:
        col_min, col_max = data[col].min(), data[col].max()

        if col_max == col_min:
            norm_data[col] = 1.0
        else:
            if col in lower_better_metrics:
                norm_data[col] = (col_max - data[col]) / (col_max - col_min)
            else:
                norm_data[col] = (data[col] - col_min) / (col_max - col_min)

    display_data = inner + (1 - inner) * norm_data
    return models, display_data

# =========================
def plot_radar(ax, models, display_data, metrics, title, legend_ncol=3):

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # =========================
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=18)
    ax.tick_params(axis='x', pad=18)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.3, 0.6,  1.0])
    ax.set_yticklabels(["0.00", "0.30", "0.60", "1.00"],
                       fontsize=16, fontweight='bold')
    ax.set_rlabel_position(20)

    ax.grid(True, linestyle='--', linewidth=0.9, alpha=0.7)
    ax.spines['polar'].set_linewidth(1.0)
    ax.spines['polar'].set_color('gray')

    # =========================
    lines = []
    for i, model in enumerate(models):
        values = display_data.loc[i, metrics].tolist()
        values += values[:1]

        line, = ax.plot(
            angles, values,
            linewidth=1.6,
            marker='o',
            markersize=3,
            linestyle='-',
            label=model
        )
        lines.append(line)

    # =========================
    legend_handles = []
    for line, model in zip(lines, models):
        legend_handles.append(Line2D(
            [0], [0],
            color=line.get_color(),
            lw=1.6,
            marker='o',
            markersize=4,
            label=model
        ))
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=legend_ncol,
        fontsize=17,
        frameon=False,

        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.3,
        labelspacing=0.3
    )

    ax.text(
        0.5, -0.45, title,
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=26
    )

# =========================
fig, axes = plt.subplots(
    1, 4,
    figsize=(24, 7.5),
    dpi=300,
    subplot_kw=dict(polar=True)
)

for idx, (ax, file_path, title) in enumerate(zip(axes, file_paths, subplot_titles)):

    models, display_data = load_and_normalize_excel(
        file_path, metrics, lower_better_metrics, inner
    )
    legend_ncol = 2 if idx == 2 else 3
    plot_radar(
        ax=ax,
        models=models,
        display_data=display_data,
        metrics=metrics,
        title=title,
        legend_ncol=legend_ncol
    )

# =========================
plt.subplots_adjust(top=0.95, bottom=0.24, wspace=0.4)
plt.savefig("figs/four_radar_charts.png", dpi=300, bbox_inches='tight')
plt.show()

