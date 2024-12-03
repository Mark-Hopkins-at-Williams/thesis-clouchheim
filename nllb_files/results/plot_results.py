import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("model_logs.csv")

# Filter out rows where 'target_lang' is English or Spanish, and 'train_scope' is 'other'
df = df[~df['target_lang'].isin(['english', 'spanish'])]
df = df[df['train_scope'] != 'other']

# Define a function to get the model identifier based on the 'train_scope' value
def get_model_identifier(row):
    if row['train_scope'] == 'multi':
        return row['model_name']
    elif row['train_scope'] == 'bi':
        return row['model_name'].split('/')[0]
    return None

# Create a new column 'model_id' for grouping
df['model_id'] = df.apply(get_model_identifier, axis=1)

# Count the number of unique languages per model_id
language_counts = df.groupby('model_id')['target_lang'].nunique().reset_index()
language_counts = language_counts[language_counts['target_lang'] >= 10]  # Only include models with at least 10 languages

# Merge the language count filter with the original dataframe
df_filtered = df[df['model_id'].isin(language_counts['model_id'])]

# Calculate the average ChrF++ score for each model_id
avg_chrf_df = df_filtered.groupby('model_id')['ChrF++'].mean().reset_index()

# Get the earliest date for each model_id
earliest_dates = df_filtered.groupby('model_id')['date'].min().reset_index()

# Merge the average ChrF++ scores with the earliest date
avg_chrf_df = avg_chrf_df.merge(earliest_dates, on='model_id')

# Sort the data by ChrF++ in descending order and select the top 10 models
top_10_models = avg_chrf_df.sort_values('ChrF++', ascending=False).head(10)

# Sort these top 10 models by the earliest date (ascending order)
top_10_models_sorted = top_10_models.sort_values('date', ascending=True)

# Plot the results for the top 10 models
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_models_sorted, x='model_id', y='ChrF++')

# Add the red horizontal line at y=30 with the label "Sheffield Standard"
plt.axhline(y=30, color='red', linestyle='--', label='Sheffield Standard')

# Add label and title
plt.xlabel('Model / Model Strategy')
plt.ylabel('Average ChrF++')
plt.title('Top 10 Models by Average ChrF++ (Sorted by Date)')

# Add the legend to display the label for the horizontal line
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout for better spacing
plt.tight_layout()

# If plt.show() does not display, you can save the plot instead
plt.savefig("top_10_avg_all_chrf_models_plot.png")

print("PNG file has been created successfully: top_10_avg_all_chrf_models_plot.png")

#########################################

# Load the CSV file
df = pd.read_csv("model_logs.csv")

# Exclude rows where 'target_lang' is English or Spanish
df_filtered = df[~df['target_lang'].isin(['english', 'spanish'])]

# For each unique language, take the row with the highest ChrF++ score
top_models = df_filtered.loc[df_filtered.groupby('target_lang')['ChrF++'].idxmax()]

# Select relevant columns (model_name, target_language, ChrF++, BLEU)
top_models = top_models[['target_lang', 'ChrF++', 'BLEU', 'model_name']]

# Rename 'target_lang' to 'target_language' for clarity
top_models = top_models.rename(columns={'target_lang': 'target_language'})

# Dictionary for comp_sheff values (you can customize this dictionary with your own values)
comp_best_dict = {
    "ashaninka": 30.5, 
    "bribri":  25.1, 
    "guarani": 36.9,  
    "quechua": 39.1,  
    "aymara": 39.1, 
    "shipibo_konibo": 35.4, 
    "chatino": 40.2, 
    "hñähñu": 14.7, 
    "nahuatl": 30.1,
    "raramuri": 20.0,
    "wixarika": 31.8
}

# Map the 'comp_sheff' column based on the dictionary
top_models['comp_best'] = top_models['target_language'].map(comp_best_dict)

# Calculate the difference compared to the comp_best for each row
top_models['comp_best_diff'] = round(top_models['ChrF++'] - top_models['comp_best'],2)

# Write the resulting dataframe to a CSV file
top_models.to_csv("top_models_per_language.csv", index=False)

print("CSV file has been created successfully: top_models_per_language_with_comp_sheff.csv")

