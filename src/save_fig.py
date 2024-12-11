import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load summary_pivot DataFrame and set the proper index
summary_pivot = pd.read_pickle('latex_tables/summary_models_table.pkl')

# Create a grouped bar plot for each language cluster
fig, ax = plt.subplots(figsize=(15, 8))

x = np.arange(len(summary_pivot.index))  # x-axis positions for each language cluster
width = 0.25  # Width of each bar

# Get model names from the DataFrame columns
models = summary_pivot.columns.levels[1]
display_names = ['NeMo', 'Phi-3', 'Aya-23']  # Simplified names for display
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot F1 scores (bin=3 and bin=5) for each model
for i, (model, display_name) in enumerate(zip(models, display_names)):
    bin_3_scores = summary_pivot[('F1(bin=3)', model)]
    bin_5_scores = summary_pivot[('F1(bin=5)', model)]
    
    # Plot bin_5 and bin_3 so that the total height is equal to bin_3
    bar_5 = ax.bar(x + i * width, bin_5_scores, width, label=f'{display_name} (bin=5)', color=colors[i], alpha=0.7, edgecolor='black')
    bar_3 = ax.bar(x + i * width, bin_3_scores - bin_5_scores, width, bottom=bin_5_scores, label=f'{display_name} (bin=3)', color=colors[i], alpha=0.4, edgecolor='black')
    
    # Add values on each bar
    for bar in bar_3:
        yval = bar.get_height() + bar.get_y()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bar_5:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=9)

# Labels, title, and legend
ax.set_xlabel('Language Cluster', fontsize=14)
ax.set_ylabel('F1 Score (%)', fontsize=14)
ax.set_title('F1 Scores by Language Cluster and Model', fontsize=16)

# Set y-axis limits from 0 to 100
ax.set_ylim(0, 70)

# Remove underscores from language cluster labels and bold 'Average' if it exists
x_labels = [label.replace('\_', ' ').replace('High german','Latvian') for label in summary_pivot.index]
x_labels = [r'$\mathbf{Average}$' if label.lower() == 'average' else label for label in x_labels]
ax.set_xticks(x + width)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)

# Place legend in the empty space
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=3, fontsize=12)

# Adjust layout and display plot
plt.tight_layout()
plt.savefig('latex_tables/summary_models_table.pdf')
plt.show()
