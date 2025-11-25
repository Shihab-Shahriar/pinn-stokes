import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_df(res_df, name):
    plot_df = res_df.melt(id_vars=["delta"],
                        value_vars=["err_rpy", "err_2b", "err_3b", "err_nbody"],
                        var_name="operator", value_name="error")
    name_map = {"err_2b": "2-body NN", "err_rpy": "RPY", "err_3b": "3-body NN", "err_nbody": "n-body NN"}
    plot_df["operator"] = plot_df["operator"].map(name_map)

    # Enforce legend order: RPY, then 2-body NN, then 3-body NN, then n-body NN
    op_order = ["RPY", "2-body NN", "3-body NN", "n-body NN"]
    plot_df["operator"] = pd.Categorical(plot_df["operator"], categories=op_order, ordered=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 5))

    # Ensure sorted by delta for cleaner lines
    plot_df_sorted = plot_df.sort_values("delta")

    # Line plot with markers for each operator
    ax = sns.lineplot(data=plot_df_sorted, x="delta", y="error", hue="operator", hue_order=op_order, marker="o")

    # Optional: explicit scatter overlay (stronger markers)
    sns.scatterplot(data=plot_df_sorted, x="delta", y="error", hue="operator", hue_order=op_order, legend=False, s=60)

    ax.set_xlabel("delta")
    ax.set_ylabel("Mean L2 error of velocity")
    ax.set_title("Grand mobility accuracy vs operator type (varying delta)")
    plt.legend(title="Mobility operator", fontsize=12, title_fontsize=12)
    plt.tight_layout()

    #save figure

    plt.savefig(f"{name}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()



res_df = pd.read_csv("data/grand_M_accuracy_varying_uniform.csv", 
                     float_precision="high", header=0, index_col=False)
plot_df(res_df, "figs/acc_grand_M_uniform")


res_df = pd.read_csv("data/grand_M_accuracy.csv", 
                     float_precision="high", header=0, index_col=False)
plot_df(res_df, "figs/acc_grand_M_fixed_delta")