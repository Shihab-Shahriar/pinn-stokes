import matplotlib.pyplot as plt
import numpy as np

def two_body(vel6d_rmse, name):

    # Calculate mean errors
    linear_error = np.mean(vel6d_rmse[:3])
    angular_error = np.mean(vel6d_rmse[3:])

    # Plotting
    labels = ['Linear Velocity', 'Angular Velocity']
    values = [linear_error, angular_error]
    colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.8, width=0.5)

    # Formatting
    plt.ylabel('RMSE (Relative)', fontsize=12)
    plt.title('ML Model Accuracy: Linear vs Angular Velocity', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=11)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'figures/ml_accuracy_{name}.png', dpi=300)
    #plt.savefig(f'figures/ml_accuracy_{name}.pdf')
    print(f"Plot saved to figures/ml_accuracy_{name}.png")


def three_acc_vs_K():
    data = [(1, 0.0017080224289159876),
        (2, 0.002433953670278601),
        (3, 0.0031706667111405054),
        (4, 0.0037350945550398054),
        (5, 0.004084014348935339),
        (6, 0.004210155591720846),
        (7, 0.004615130375397149),
        (8, 0.004642590949376501),
        (9, 0.004691374923923932),
        (10, 0.005045808821119911)
    ]

    x, y = zip(*data)

    plt.figure(figsize=(8, 6))

    # Bar plot
    bars = plt.bar(x, y, color='#1f77b4', alpha=0.7, label='Error')

    # Line plot over center of each bar head
    plt.plot(x, y, color='red', marker='o', linestyle='-', linewidth=2, label='Trend')

    # Formatting
    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Accuracy vs K', fontsize=14)
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/accuracy_vs_k.png', dpi=300)
    print("Plot saved to figures/accuracy_vs_k.png")


if __name__ == "__main__":
    # Example RMSE values for two-body interaction (linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
    vel_2b = [0.022281, 0.023004, 0.028248, 0.029638, 0.038296, 0.034684]

    vel_3b = [2.32376, 2.36029, 2.30545, 8.38881, 8.34584, 8.54198]

    two_body(vel_2b, "2b")
    two_body(vel_3b, "3b")
    three_acc_vs_K()