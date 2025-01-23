import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the folder names (replace with actual folder names if different)
folders = ["bilingual", "parallel", "not_parallel"]

# Filter argument for 'tgt' column
allowed_tgt_values = []  # Replace with desired target values

# Initialize data structures
baseline_folder = folders[0]
folder_names = []
score_differences = []
error_bars = []

# Read baseline data
baseline_file = os.path.join(baseline_folder, "scores.csv")
baseline_df = pd.read_csv(baseline_file)
if allowed_tgt_values != []:
    baseline_df = baseline_df[baseline_df["tgt"].isin(allowed_tgt_values)]
# Group by 'model' and calculate the mean CHRF score
baseline_scores = baseline_df.groupby("model")["chrf"].mean()

print('baseline', baseline_scores)
# Iterate over each folder
for folder in folders:
    file_path = os.path.join(folder, "scores.csv")
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        if allowed_tgt_values != []:
            df = df[df["tgt"].isin(allowed_tgt_values)]

        if not df.empty:
            # Group by 'model' and calculate the mean CHRF score
            averaged_scores = df.groupby("model")["chrf"].mean()
            print(averaged_scores)
            
            # Align with baseline scores and calculate differences
            aligned_scores = averaged_scores.reindex(baseline_scores.index)
            #print(f"Aligned scores for folder {folder}:\n{aligned_scores}")
            differences = aligned_scores - baseline_scores
            #print(f"Differences for folder {folder}:\n{differences}")

            # Calculate mean and standard deviation of differences
            mean_diff = differences.mean()
            std_diff = differences.std()

            # Append results
            folder_names.append(folder)
            score_differences.append(mean_diff)
            error_bars.append(std_diff)
        else:
            print(f"No matching rows in {file_path} after filtering by 'tgt'.")
    else:
        print(f"File not found: {file_path}")

# Create the bar graph
x_positions = np.arange(len(folder_names))

plt.figure(figsize=(8, 6))
plt.bar(x_positions, score_differences, yerr=error_bars, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')

# Add labels and title
plt.xticks(x_positions, folder_names, fontsize=10)
plt.ylabel("Difference in CHRF++ Score", fontsize=12)
plt.xlabel("Train Type", fontsize=12)
plt.title("Differences in CHRF++ Scores Relative to Bilingual", fontsize=14)

# Add value labels to the top and bottom of the error bars
for x, diff, err in zip(x_positions, score_differences, error_bars):
    plt.text(x, diff + err + 0.01, f"{diff + err:.2f}", ha='center', fontsize=9, color='green')
    plt.text(x, diff - err - 0.03, f"{diff - err:.2f}", ha='center', fontsize=9, color='red')

# Save the plot to a file
plt.tight_layout()
plt.savefig("bar_graph.png")

# Display the plot
plt.show()
