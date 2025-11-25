import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.mob_op_2b_combined import NNMob as TwoBodyNNMob
from src.mob_op_3body import NNMob3B
from src.mob_op_nbody import Mob_Op_Nbody
from benchmarks.cluster import reference_data_generation, uniform_data_generation
from src.gpu_nbody_mob import Mob_Nbody_Torch

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

mob_nbody = Mob_Op_Nbody(
    shape=shape,
    self_nn_path=self_path,
    two_nn_path=two_body,
    nbody_nn_path="data/models/nbody_pinn_b1.pt",
    nn_only=False,
    rpy_only=False,
    switch_dist=6.0,
)

# mob_nbody = Mob_Nbody_Torch(
#     shape=shape,
#     self_nn_path=self_path,
#     two_nn_path=two_body,
#     nbody_nn_path="data/models/nbody_pinn_b1.pt",
#     nn_only=False,
#     rpy_only=False,
#     switch_dist=6.0,
# )


numParticles = 20
results = []
deltas = [.1, .2, .4, .5, .6, .8, 1.0,1.5, 2.0, 3.0] #At 4.0 & 5.0 everyone equal
deltas = [.2, .4, .5, .6, .8, 1.0, 1.5, 2.0]
vol_fracs = [.05, .1, .15, .2, .22][::-1]

uniform_config = True
iters = vol_fracs if uniform_config else deltas
name = "grand_M_accuracy" if not uniform_config else "grand_M_accuracy_varying_uniform"

print("Uniform config:", uniform_config)
for delta in iters:
    print("Delta:", delta)
    
    #df = reference_data_generation(shape, delta, numParticles)
    if uniform_config:
        df = pd.read_csv(f"tmp/uniform_{delta}_{numParticles}.csv", float_precision="high",
                         header=0, index_col=False)
    else:
        df = pd.read_csv(f"tmp/reference_{delta}_{numParticles}.csv", float_precision="high",
                         header=0, index_col=False)
        # df = pd.read_csv(f"tmp/reference_sphere_{delta}.csv", float_precision="high",
        #             header=0, index_col=False)
        print("num P:", df.shape[0])


    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values
    forces = df[["f_x","f_y","f_z","t_x","t_y","t_z"]].values
    velocity = df[["v_x","v_y","v_z","w_x","w_y","w_z"]].values

    v_2b = mob_2b.apply(config, forces, viscosity=1.0)
    v_rpy = mob_rpy.apply(config, forces, viscosity=1.0)
    v_3b = mob_3b.apply(config, forces, viscosity=1.0)
    v_nbody = mob_nbody.apply(config, forces, viscosity=1.0)

    ND = 3
    err_2b = np.linalg.norm(velocity[:,:ND] - v_2b[:,:ND], axis=1).mean()
    err_rpy = np.linalg.norm(velocity[:,:ND] - v_rpy[:,:ND], axis=1).mean()
    err_3b = np.linalg.norm(velocity[:,:ND] - v_3b[:,:ND], axis=1).mean()
    err_nbody = np.linalg.norm(velocity[:,:ND] - v_nbody[:,:ND], axis=1).mean()

    results.append({
        "delta": float(delta),
        "err_rpy": float(err_rpy),
        "err_2b": float(err_2b),
        "err_3b": float(err_3b),
        "err_nbody": float(err_nbody),
    })

    print(f"errors -> 2b: {err_2b:.6e}," 
            f"RPY: {err_rpy:.6e}, "
            f"3b: {err_3b:.6e}, "
            f"nbody: {err_nbody:.6e}")

# Convert to DataFrame with explicit column order: RPY, then 2b, then 3b, then nbody
res_df = pd.DataFrame(results, columns=["delta", "err_rpy", "err_2b", "err_3b", "err_nbody"])
res_df = res_df.round(5)
res_df.to_csv(f"{name}.csv", index=False)
print(f"Saved results to {name}.csv")

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
plt.savefig(f"{name}.png", dpi=300)
plt.show()