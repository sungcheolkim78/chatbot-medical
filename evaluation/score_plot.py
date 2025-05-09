import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# Read all CSV files that start with llm_score
csv_files = glob.glob('evaluation/chatbot_results/llm_score_*.csv')
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all dataframes
df = pd.concat(dfs, ignore_index=True)

# Set the style for better visualization
plt.figure(figsize=(15, 8))

# Create box plots for each metric, grouped by model_name
metrics = ['correctness', 'response_time', 'style']  # Reordered metrics
df_melted = pd.melt(df, id_vars=['model_name'], value_vars=metrics, 
                    var_name='Metric', value_name='Score')

# Sort the data by model_name alphabetically
df_melted = df_melted.sort_values('model_name')
print(df_melted)

# Create the box plot with increased spacing
sns.boxplot(x='Metric', y='Score', hue='model_name', data=df_melted, width=0.7, dodge=True)

# Customize the plot
plt.title('Distribution of Metrics by Model', fontsize=14, pad=20)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)

# Add legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('evaluation/chatbot_results/metrics_boxplot_by_model.png', dpi=300, bbox_inches='tight')
plt.close() 