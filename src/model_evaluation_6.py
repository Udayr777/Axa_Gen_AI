import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Load Prediction & Ground Truth Data
file_predictions = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\balanced_final_classified_results.csv"
file_ground_truth = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\balanced_ground_truth_dataset.csv"

# Read Data
df_predictions = pd.read_csv(file_predictions)
df_ground_truth = pd.read_csv(file_ground_truth)

# Standardizing Column Names
df_predictions = df_predictions.rename(columns={"Sentiment": "Sentiment_predicted"})
df_ground_truth = df_ground_truth.rename(columns={"Sentiment_actual": "Sentiment"})

# Merge on 'Customer Statement'
df_merged = df_predictions.merge(df_ground_truth, on="Customer Statement", how="inner")

# Extract labels
y_true = df_merged["Sentiment"]
y_pred = df_merged["Sentiment_predicted"]

# Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

# Print Key Evaluation Metrics
print("\n**Evaluation Metrics (Fine-Tuned Model)**")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Classification Report
print("\n**Detailed Classification Report**")
print(classification_report(y_true, y_pred, zero_division=1))

# Confusion Matrix
plt.figure(figsize=(7, 6))
conf_matrix = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative", "Neutral"])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Positive", "Negative", "Neutral"], yticklabels=["Positive", "Negative", "Neutral"])
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix - Fine-Tuned Model")
plt.show()

print("\n**Model Evaluation Completed!**")
