import pandas as pd
from sklearn.utils import resample

# Load the Ground Truth Dataset
ground_truth_path = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\ground_truth_dataset.csv"
df_ground_truth = pd.read_csv(ground_truth_path)

# Check Initial Label Distribution
label_counts = df_ground_truth["Sentiment_actual"].value_counts()
print("\nOriginal Label Distribution:\n", label_counts)

# 3: Separate Classes
df_positive = df_ground_truth[df_ground_truth["Sentiment_actual"] == "Positive"]
df_neutral = df_ground_truth[df_ground_truth["Sentiment_actual"] == "Neutral"]
df_negative = df_ground_truth[df_ground_truth["Sentiment_actual"] == "Negative"]

# Adaptive Sampling
# Setting the target size to the class with the lowest count
target_size = len(df_neutral)

df_positive_resampled = resample(df_positive, replace=True, n_samples=target_size, random_state=42)
df_negative_resampled = resample(df_negative, replace=True, n_samples=target_size, random_state=42)

# Merge Balanced Dataset
df_balanced_ground_truth = pd.concat([df_positive_resampled, df_neutral, df_negative_resampled])
df_balanced_ground_truth = df_balanced_ground_truth.sample(frac=1, random_state=42)  # Shuffle dataset

# Verify the New Label Distribution
balanced_label_counts = df_balanced_ground_truth["Sentiment_actual"].value_counts()
print("\nBalanced Label Distribution:\n", balanced_label_counts)

# Save the New Balanced Ground Truth Dataset
balanced_ground_truth_path = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\balanced_ground_truth_dataset.csv"
df_balanced_ground_truth.to_csv(balanced_ground_truth_path, index=False)

print(f"\nBalanced ground truth dataset saved at: {balanced_ground_truth_path}")
