import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# ---------------------------------------------------------
# 1. Data Setup (No Pandas)
# ---------------------------------------------------------

# Raw data extracted from logs
# Format: {'method': str, 'batch_size': int, 'throughput_msps': float}
# Note: 'msps' = Million samples per second

raw_data = [
    # --- NeuralNet Data (CUDA) ---
    # Excluded first two points (16384: 0.19, 24064: 0.26) due to 
    # JIT compilation overhead/warmup which distorts the scale.
    {'method': 'NeuralNet', 'batch_size': 32768,   'throughput_msps': 86.22},
    {'method': 'NeuralNet', 'batch_size': 65536,   'throughput_msps': 178.71},
    {'method': 'NeuralNet', 'batch_size': 131072,  'throughput_msps': 330.64},
    {'method': 'NeuralNet', 'batch_size': 262144,  'throughput_msps': 413.18},
    {'method': 'NeuralNet', 'batch_size': 524288,  'throughput_msps': 452.26},
    {'method': 'NeuralNet', 'batch_size': 1048576, 'throughput_msps': 463.81},
    {'method': 'NeuralNet', 'batch_size': 2097152, 'throughput_msps': 486.87},
    {'method': 'NeuralNet', 'batch_size': 4194304, 'throughput_msps': 495.37},

    # --- RPY Data (Float32) ---
    # {'method': 'RPY', 'batch_size': 16384,   'throughput_msps': 167.96},
    # {'method': 'RPY', 'batch_size': 24064,   'throughput_msps': 282.52},
    {'method': 'RPY', 'batch_size': 32768,   'throughput_msps': 358.38},
    {'method': 'RPY', 'batch_size': 65536,   'throughput_msps': 717.85},
    {'method': 'RPY', 'batch_size': 131072,  'throughput_msps': 838.51},
    {'method': 'RPY', 'batch_size': 262144,  'throughput_msps': 909.03},
    {'method': 'RPY', 'batch_size': 524288,  'throughput_msps': 955.34},
    {'method': 'RPY', 'batch_size': 1048576, 'throughput_msps': 979.31},
    {'method': 'RPY', 'batch_size': 2097152, 'throughput_msps': 991.14},
    {'method': 'RPY', 'batch_size': 4194304, 'throughput_msps': 1001.03},
]

# Process data: Separate into lists for plotting
# Converting Msps to Samples/sec (x 1,000,000)
batch_sizes = []
throughputs = []
methods = []

for entry in raw_data:
    batch_sizes.append(entry['batch_size'])
    throughputs.append(entry['throughput_msps'] * 1e6) 
    methods.append(entry['method'])

# ---------------------------------------------------------
# 2. Plotting (Seaborn + Matplotlib)
# ---------------------------------------------------------

# Set high-quality style
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.figure(figsize=(8, 6))

# Define custom colors and markers
palette = {"NeuralNet": "#D62728", "RPY": "#1F77B4"} # Red and Blue
markers = {"NeuralNet": "o", "RPY": "s"}

# Create the plot
ax = sns.lineplot(
    x=batch_sizes,
    y=throughputs,
    hue=methods,
    style=methods,
    markers=markers,
    markersize=9,     # Slightly larger markers for visibility
    dashes=False,     # Solid lines for both
    palette=palette,
    linewidth=2.5
)

# ---------------------------------------------------------
# 3. Formatting
# ---------------------------------------------------------

# Logarithmic X Axis
ax.set_xscale("log", base=2)

# Labels
ax.set_xlabel("Batch Size ($N$)", fontsize=14, labelpad=10)
ax.set_ylabel("Throughput (Samples/s)", fontsize=14, labelpad=10)
ax.set_title("Throughput Comparison: RPY vs NeuralNet", fontsize=16, pad=15)

# Grid setup
ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.8)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

# Force scientific notation on Y axis if not automatic
# (Matplotlib usually handles this well, but this ensures paper style)
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Legend customization
plt.legend(title=None, fontsize=12, frameon=True, loc="upper left")

# Despine (remove top and right borders)
sns.despine()

# ---------------------------------------------------------
# 4. Save Output
# ---------------------------------------------------------
output_file = "figs/throughput_plot.pdf"
plt.tight_layout()
plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')

print(f"Plot saved successfully as {output_file}")
plt.show()