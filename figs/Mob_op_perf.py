import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_comparison():
    # ---------------------------------------------------------
    # 1. DATA CONFIGURATION
    # ---------------------------------------------------------
    
    # The operators tested
    operators = ['M_2b', 'M_rpy', 'M_nbody']

    # --- Real Data (Nvidia RTX 4060) ---
    # Extracted from your log
    gpu1_name = "Nvidia RTX 4060"
    gpu1_means = [22.61, 21.28, 31.92]  # Time in ms
    gpu1_std   = [0.28, 0.46, 0.27]    # Standard Deviation

    # --- Dummy Data (Nvidia H200) ---
    # Assumptions: H200 is significantly faster (lower ms). 
    # We create dummy data roughly 8-10x faster.
    gpu2_name = "Nvidia H200"
    gpu2_means = [2.60, 2.48, 5.5]    
    gpu2_std   = [0.05, 0.03, 0.12]    # Lower variance on high-end hardware

    # ---------------------------------------------------------
    # 2. PLOTTING SETUP
    # ---------------------------------------------------------
    
    # X-axis locations for the groups
    x = np.arange(len(operators))
    
    # Width of each bar
    width = 0.35  

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------------------------------------------------------
    # 3. CREATE BARS
    # ---------------------------------------------------------
    
    # Plot GPU 1 (4060)
    rects1 = ax.bar(x - width/2, gpu1_means, width, 
                    yerr=gpu1_std, label=gpu1_name, 
                    color='#4c72b0', capsize=5, alpha=0.9, edgecolor='black')

    # Plot GPU 2 (H200)
    rects2 = ax.bar(x + width/2, gpu2_means, width, 
                    yerr=gpu2_std, label=gpu2_name, 
                    color='#55a868', capsize=5, alpha=0.9, edgecolor='black')

    # ---------------------------------------------------------
    # 4. FORMATTING AND LABELS
    # ---------------------------------------------------------
    
    ax.set_ylabel('Mean Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Mobility Operator Performance: RTX 4060 vs H200', fontsize=14, pad=20)
    
    # Set x-axis ticks to be the operator names
    ax.set_xticks(x)
    ax.set_xticklabels(operators, fontsize=11)
    
    # Add a grid for easier reading (behind the bars)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Add Legend
    ax.legend(fontsize=11)

    # ---------------------------------------------------------
    # 5. AUTO-LABELING VALUE FUNCTION
    # ---------------------------------------------------------
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # ---------------------------------------------------------
    # 6. DISPLAY/SAVE
    # ---------------------------------------------------------
    
    plt.tight_layout()
    
    # Save the plot (optional)
    plt.savefig('figs/gpu_benchmark_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Plot saved as 'figs/gpu_benchmark_comparison.pdf'")

    # Show the plot
    plt.show()


def plot_h200_problem_sizes():
    # ---------------------------------------------------------
    # 1. DATA CONFIGURATION
    # ---------------------------------------------------------

    problem_sizes = ['N=1600', 'N=3200', 'N=6400']
    operators = [ 'NNMobTorch_rpy', 'NNMobTorch', 'NNMob_GPU_Nbody']
    labels  = {"NNMobTorch_rpy": "M_rpy",
               "NNMobTorch": "M_2b",
               "NNMob_GPU_Nbody": "M_nbody"}

    means = {
        'NNMob_GPU_Nbody': [5.50, 20.52, 50.97],
        'NNMobTorch_rpy': [2.48, 11.06, 48.01],
        'NNMobTorch': [2.60, 11.51, 48.85],
    }

    stds = {
        'NNMob_GPU_Nbody': [0.36, 3.86, 0.34],
        'NNMobTorch_rpy': [0.02, 0.12, 0.08],
        'NNMobTorch': [0.01, 0.14, 0.10],
    }

    colors = {
        'NNMob_GPU_Nbody': '#4c72b0',
        'NNMobTorch_rpy': '#55a868',
        'NNMobTorch': '#dd8452',
    }

    # ---------------------------------------------------------
    # 2. PLOTTING SETUP
    # ---------------------------------------------------------

    x = np.arange(len(problem_sizes))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------------------------------------------------------
    # 3. CREATE BARS
    # ---------------------------------------------------------

    rect_groups = []
    for idx, operator in enumerate(operators):
        offset = (idx - (len(operators) - 1) / 2) * (width + 0.03)
        bars = ax.bar(
            x + offset,
            means[operator],
            width,
            yerr=stds[operator],
            label=labels[operator],
            color=colors[operator],
            capsize=5,
            alpha=0.9,
            edgecolor='black',
        )
        rect_groups.append(bars)

    # ---------------------------------------------------------
    # 4. FORMATTING AND LABELS
    # ---------------------------------------------------------

    ax.set_ylabel('Mean Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('H200 Performance Across Problem Sizes', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(problem_sizes, fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)


    # ---------------------------------------------------------
    # 5. AUTO-LABELING VALUE FUNCTION
    # ---------------------------------------------------------
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if np.isnan(height):
                continue
            ax.annotate(
                f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )

    for rects in rect_groups:
        autolabel(rects)

    # ---------------------------------------------------------
    # 6. DISPLAY/SAVE
    # ---------------------------------------------------------

    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig('figs/h200_problem_size_performance.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Plot saved as 'figs/h200_problem_size_performance.pdf'")
    plt.show()


if __name__ == "__main__":
    plot_benchmark_comparison()
    #plot_h200_problem_sizes()