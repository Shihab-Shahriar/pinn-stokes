from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_m_accuracy_nbody_diff_sizes(
	csv_path="data/M_accuracy_nbody_diff_sizes.csv",
	output_dir="figures",
	target_metric = "avg_rel_rmse"
):
	df = pd.read_csv(csv_path)
	df = df[df["volume_fraction"].isin([0.05, 0.1, .15])]

	df = df[df["num_particles"] <= 140]

	sns.set(style="whitegrid")
	plt.figure(figsize=(8, 5))

	warm_colorblind = ["#E69F00", "#D55E00", "#F0E442", "#CC79A7"]
	ax = sns.lineplot(
		data=df,
		x="num_particles",
		y=target_metric,
		hue="volume_fraction",
		marker="o",
		palette=warm_colorblind,
	)

	ax.set_xlabel("N")
	ax.set_ylabel("Relative RMSE Error")
	ax.set_title("Nemo Accuracy vs Particle Count")
	ax.set_ybound(lower=1e-6, upper=df.max()[target_metric] * 1.5)
	plt.legend(title="Volume fraction")
	plt.tight_layout()

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_prefix = output_dir / "M_accuracy_nbody_diff_sizes"
	plt.savefig(f"{output_prefix}_{target_metric}.png", dpi=600)
	#plt.savefig(f"{output_prefix}.pdf", dpi=600)
	plt.close()


def plot_grand_m_accuracy_fixed_N(
	csv_path="data/grand_M_acc_uniform_fixed_N.csv",
	output_dir="figures",
	N=40
):
	df = pd.read_csv(csv_path)
	df = df[df["num_particles"] == N]
	if "mfs_coarse" in df.columns:
		df = df.drop(columns=["mfs_coarse"])
	
	# Melt the dataframe to have Method and Error columns for plotting
	df_melted = df.melt(
		id_vars=["volume_fraction", "num_particles"],
		var_name="Method",
		value_name="Error"
	)
	df_melted["Error"] /= 100

	sns.set(style="whitegrid")
	plt.figure(figsize=(8, 5))

	ax = sns.lineplot(
		data=df_melted,
		x="volume_fraction",
		y="Error",
		hue="Method",
		marker="o",
	)

	ax.set_xlabel("Volume Fraction")
	ax.set_ylabel("Relative RMSE Error")
	ax.set_title(f"Accuracy vs Volume Fraction (N={N})")
	#plt.yscale('log')
	plt.legend(title="Method")
	plt.tight_layout()

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_dir / f"grand_M_accuracy_N{N}.png", dpi=600)
	plt.close()



def random_vs_sedimenting_force(output_dir="figures"):
	"""
	For P=50.
	"""
	sediment = {
		.05: 1.375,
		.1: 2.63,
		.15: 3.88,
		.20: 5.05
    }
	random = {
		.05: 2.49,
		.1: 5.1,
		.15: 8.40,
		.20: 10.17
    }

	data = []
	for phi, val in sediment.items():
		data.append({"Volume Fraction": phi, "Error": val, "Force Type": "Sedimenting"})
	for phi, val in random.items():
		data.append({"Volume Fraction": phi, "Error": val, "Force Type": "Random"})
	
	df = pd.DataFrame(data)

	sns.set(style="whitegrid")
	plt.figure(figsize=(8, 5))

	ax = sns.lineplot(
		data=df,
		x="Volume Fraction",
		y="Error",
		hue="Force Type",
		marker="o",
	)
	
	ax.set_ylabel("Relative RMSE Error (%)")
	ax.set_title("Accuracy vs Volume Fraction (N=50)")
	plt.tight_layout()

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_dir / "random_vs_sedimenting.png", dpi=600)
	plt.close()

if __name__ == "__main__":
	# plot_m_accuracy_nbody_diff_sizes(target_metric="avg_rel_rmse")
	# plot_grand_m_accuracy_fixed_N(N=50)
	random_vs_sedimenting_force()
