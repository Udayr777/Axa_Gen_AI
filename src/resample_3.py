import pandas as pd
from sklearn.utils import resample

# Load the Dataset
file_path = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\final_classified_results.csv"
df = pd.read_csv(file_path)

# Verify Original Label Distribution
print("\n### Original Sentiment Distribution ###")
print(df["Sentiment"].value_counts())

# Separate Sentiment Classes
df_positive = df[df["Sentiment"] == "Positive"]
df_neutral = df[df["Sentiment"] == "Neutral"]
df_negative = df[df["Sentiment"] == "Negative"]

# Upsample Positive and Negative to Match Neutral (859 samples)
df_positive_upsampled = resample(df_positive, replace=True, n_samples=859, random_state=42)
df_negative_upsampled = resample(df_negative, replace=True, n_samples=859, random_state=42)

# Combine & Shuffle the Dataset
df_balanced = pd.concat([df_positive_upsampled, df_neutral, df_negative_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle dataset

# Verify New Label Distribution
print("\n### Balanced Sentiment Distribution ###")
print(df_balanced["Sentiment"].value_counts())

# Convert Sentiment Labels to Numeric Format
label_mapping = {"Positive": 2, "Neutral": 1, "Negative": 0}
df_balanced["label"] = df_balanced["Sentiment"].map(label_mapping)

# Save the Balanced Dataset for Fine-Tuning
balanced_file_path = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\balanced_final_classified_results.csv"
df_balanced.to_csv(balanced_file_path, index=False)

print(f"\nBalanced dataset saved at: {balanced_file_path}")