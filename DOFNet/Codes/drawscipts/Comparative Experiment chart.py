import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

excel_path = r"comparative_results.xlsx"
sheet_name = 0
save_path = "figs/tradeoff_1x3.png"

datasets = ["DFID", "OpenDF", "Caries"]
subplot_labels = ["(a)", "(b)", "(c)"]
target_metric = "ACC"

annotate_models = False

# =========================
def flatten_columns(columns):
    flat_cols = []
    for col in columns:
        if isinstance(col, tuple):
            parts = []
            for x in col:
                s = str(x).strip()
                if s and s != "nan" and not s.startswith("Unnamed"):
                    parts.append(s)
            flat_cols.append("_".join(parts))
        else:
            flat_cols.append(str(col).strip())
    return flat_cols

def norm_text(s):
    s = str(s).strip()
    s = s.replace("\n", "")
    s = s.replace(" ", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("%", "")
    return s.lower()

def find_col(columns, keywords):
    keywords = [norm_text(k) for k in keywords]
    for c in columns:
        cc = norm_text(c)
        if all(k in cc for k in keywords):
            return c
    raise KeyError(f"找不到列 {keywords}，当前列名为：\n{columns}")

df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])
df_raw.columns = flatten_columns(df_raw.columns)

model_col = find_col(df_raw.columns, ["model"])
params_col = find_col(df_raw.columns, ["params"])
flops_col = find_col(df_raw.columns, ["flops"])

records = []
for _, row in df_raw.iterrows():
    model = str(row[model_col]).strip()
    params = row[params_col]
    flops = row[flops_col]

    for ds in datasets:
        rec = {
            "Model": model,
            "Params": params,
            "FLOPs": flops,
            "Dataset": ds
        }
        try:
            metric_col = find_col(df_raw.columns, [ds, target_metric])
            rec[target_metric] = row[metric_col]
        except KeyError:
            rec[target_metric] = np.nan
        records.append(rec)

df = pd.DataFrame(records)

for col in ["Params", "FLOPs", target_metric]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Model", "Params", "FLOPs", target_metric]).copy()

# =========================
def get_family(model_name):
    m = model_name.lower().strip()

    if "dofnet" in m:
        return "DOFNet"
    elif "resnet" in m:
        return "ResNet"
    elif "densenet" in m:
        return "DenseNet"
    elif "mobilenetv3" in m:
        return "MobileNetV3"
    elif re.match(r"vit", m):
        return "ViT"
    elif "swin" in m:
        return "Swin"
    elif "mltrmr" in m:
        return "MLTrMR"
    elif "fusiondentnet" in m:
        return "FusionDentNet"
    elif "ld2net" in m:
        return "LD2Net"
    elif "diffmic" in m:
        return "DiffMIC"
    else:
        return model_name

df["Family"] = df["Model"].apply(get_family)
df["Is_Ours"] = df["Family"] == "DOFNet"

# =========================
family_colors = {
    "DOFNet":        "#E41A1C",
    "ResNet":        "#377EB8",
    "DenseNet":      "#008000",
    "MobileNetV3":   "#ffa500",
    "ViT":           "#984EA3",
    "Swin":          "#F39C12",
    "MLTrMR":        "#17BECF",
    "FusionDentNet": "#E377C2",
    "DiffMIC": "#00ffff",
    "LD2Net": "#a52a2a",
}

# =========================
flops_max = df["FLOPs"].max()

def compress_flops(x):
    x = np.asarray(x, dtype=float)
    return 100 * np.log1p(x) / np.log1p(flops_max)

df["FLOPs_compressed"] = compress_flops(df["FLOPs"])

def scatter_size_from_flops(flops):
    return 120

def legend_markersize_from_flops(flops):
    return 7.5

# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)

for ax, ds, sublabel in zip(axes, datasets, subplot_labels):
    sub = df[df["Dataset"] == ds].copy()
    legend_handles = []

    for family in sub["Family"].unique():
        fam_df = sub[sub["Family"] == family].copy()
        fam_df = fam_df.sort_values(by="FLOPs")

        color = family_colors.get(family, "#7F7F7F")
        marker = "*" if family == "DOFNet" else "^"
        linestyle = "-" if family == "DOFNet" else ":"

        x = fam_df["FLOPs_compressed"].values
        y = fam_df[target_metric].values

        if len(fam_df) >= 2:
            ax.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                alpha=0.95,
                zorder=2
            )

        for _, r in fam_df.iterrows():
            s = scatter_size_from_flops(r["FLOPs"])
            ax.scatter(
                r["FLOPs_compressed"],
                r[target_metric],
                s=s,
                marker=marker,
                color=color,
                edgecolors="none",
                zorder=3
            )

            if annotate_models:
                if family == "DOFNet" or r["FLOPs"] >= 150:
                    ax.annotate(
                        r["Model"],
                        (r["FLOPs_compressed"], r[target_metric]),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=8,
                        color=color
                    )

        for _, r in fam_df.iterrows():
            ms = legend_markersize_from_flops(r["FLOPs"])
            handle = Line2D(
                [0], [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=ms,
                linewidth=1.6,
                label=r["Model"]
            )
            legend_handles.append(handle)

    ax.set_xlim(-2, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100],)
    ax.set_xlabel("FLOPs", fontsize=18)
    ax.tick_params(
        axis='both',
        labelsize=16
    )

    y_min = sub[target_metric].min() - 2
    y_max = sub[target_metric].max() + 2
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("ACC (%)", fontsize=16)

    ax.grid(True, linestyle="-", alpha=0.35)

    if ds == "DFID":
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=True,
            fontsize=9,
            borderaxespad=0.3,
            ncol=1,
            labelspacing=0.15,
            handletextpad=0.3,
            borderpad=0.25,
        )

    else:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=True,
            fontsize=10,
            borderaxespad=0.3,
            ncol=1,
            labelspacing=0.15,
            handletextpad=0.3,
            borderpad=0.25,
        )

    ax.text(
        0.5, -0.2, sublabel,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=24,

    )

# =========================
plt.tight_layout()
plt.subplots_adjust(bottom=0.20, wspace=0.18)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"saving: {save_path}")