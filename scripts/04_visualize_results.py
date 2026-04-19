import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("Generating portfolio visualizations...")

# 1. The Data
# Replace this array with the exact 20 percentages from your final terminal output
# I have pre-filled the ones you pasted (Subjects 9-20) and estimated 1-8.
lda_scores = [
    61.2, 58.4, 65.1, 59.8, 62.3, 54.5, 68.9, 60.1, # Replace 1-8 with your actual numbers
    71.11, 52.22, 47.78, 44.44, 66.67, 71.11, 75.56, 55.56, 42.22, 66.67, 64.44, 48.89
]
subjects = [f"S{i}" for i in range(1, 21)]

# Baseline data
svm_average = 49.94
lda_average = np.mean(lda_scores)

# Set the academic aesthetic
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- PLOT 1: Intra-Subject Variance (Bar Chart) ---
# Create a dataframe for easy plotting
df_subjects = pd.DataFrame({'Subject': subjects, 'Accuracy (%)': lda_scores})
df_subjects = df_subjects.sort_values(by='Accuracy (%)', ascending=False) # Sort for impact

# Color bars based on performance (>60% is good, <50% is failing)
colors = ['#2ecc71' if score >= 60 else '#e74c3c' if score < 50 else '#f1c40f' for score in df_subjects['Accuracy (%)']]

sns.barplot(data=df_subjects, x='Subject', y='Accuracy (%)', palette=colors, ax=axes[0])
axes[0].axhline(50, color='black', linestyle='--', linewidth=1.5, label='Random Chance (50%)')
axes[0].axhline(lda_average, color='blue', linestyle=':', linewidth=2, label=f'LDA Average ({lda_average:.1f}%)')

axes[0].set_title('CSP-LDA Intra-Subject Decoding Accuracy', fontsize=14, pad=15)
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(loc='upper right')
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_xlabel('Subject ID', fontsize=12)

# --- PLOT 2: Model Architecture Comparison (Bar Chart) ---
models = ['Generalized SVM\n(Cross-Subject)', 'Optimized CSP-LDA\n(Intra-Subject)']
averages = [svm_average, lda_average]

sns.barplot(x=models, y=averages, palette=['#95a5a6', '#3498db'], ax=axes[1], width=0.5)
axes[1].axhline(50, color='black', linestyle='--', linewidth=1.5)

axes[1].set_title('Model Architecture Comparison', fontsize=14, pad=15)
axes[1].set_ylim(0, 100)
axes[1].set_ylabel('Average Accuracy (%)', fontsize=12)

# Add value labels on top of the bars
for i, v in enumerate(averages):
    axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save the plot as a high-resolution PNG for your GitHub README
output_file = "BCI_Results_Figure.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Success! High-resolution graph saved as: {output_file}")