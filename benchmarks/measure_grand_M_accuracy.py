import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.mob_op_2b_combined import NNMob as TwoBodyNNMob
from src.mob_op_3body import NNMob3B
from benchmarks.cluster import reference_data_generation, uniform_data_generation

# None of them using self interaction NN model
shape = "sphere"
self_path = "data/models/self_interaction_model.pt"
two_body = "data/models/two_body_combined_model.pt"
mob_2b = TwoBodyNNMob(shape, self_path, two_body, 
            nn_only=False, rpy_only=False)

mob_rpy = TwoBodyNNMob(shape, self_path, two_body, 
            nn_only=False, rpy_only=True)

three_body = "data/models/3body_cross.pt"

mob_3b = NNMob3B(
    shape=shape,
    self_nn_path=self_path,
    two_nn_path=two_body,
    three_nn_path=three_body,
    nn_only=False,
    rpy_only=False,
    switch_dist=6.0,
    triplet_cutoff=6.0,
)

numParticles = 10
results = []
deltas = [.1, .2, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
for delta in deltas:
    print("Delta:", delta)
    df = reference_data_generation(shape, delta, numParticles)
    # uniform_df = uniform_data_generation(shape, delta, numParticles)

    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values
    forces = df[["f_x","f_y","f_z","t_x","t_y","t_z"]].values
    velocity = df[["v_x","v_y","v_z","w_x","w_y","w_z"]].values

    v_2b = mob_2b.apply(config, forces, viscosity=1.0)
    v_rpy = mob_rpy.apply(config, forces, viscosity=1.0)
    v_3b = mob_3b.apply(config, forces, viscosity=1.0)

    err_2b = np.linalg.norm(velocity - v_2b, axis=1).mean()
    err_rpy = np.linalg.norm(velocity - v_rpy, axis=1).mean()
    err_3b = np.linalg.norm(velocity - v_3b, axis=1).mean()

    results.append({
        "delta": float(delta),
        "err_rpy": float(err_rpy),
        "err_2b": float(err_2b),
        "err_3b": float(err_3b),
    })

    print(f"  errors -> 2b: {err_2b:.6e}, RPY: {err_rpy:.6e}, 3b: {err_3b:.6e}")

# Convert to DataFrame with explicit column order: RPY, then 2b, then 3b
res_df = pd.DataFrame(results, columns=["delta", "err_rpy", "err_2b", "err_3b"])
res_df.to_csv("grand_M_accuracy.csv", index=False)
print("Saved results to grand_M_accuracy.csv")

# Plot with seaborn directly from res_df
plot_df = res_df.melt(id_vars=["delta"],
                      value_vars=["err_rpy", "err_2b", "err_3b"],
                      var_name="operator", value_name="error")
name_map = {"err_2b": "2-body NN", "err_rpy": "RPY", "err_3b": "3-body NN"}
plot_df["operator"] = plot_df["operator"].map(name_map)

# Enforce legend order: RPY, then 2-body NN, then 3-body NN
op_order = ["RPY", "2-body NN", "3-body NN"]
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
plt.savefig("grand_M_accuracy.png", dpi=300)
plt.show()