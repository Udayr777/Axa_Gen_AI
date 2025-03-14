import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Load Prediction Data
file_predictions = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\balanced_final_classified_results.csv"
df_predictions = pd.read_csv(file_predictions)

# Debugging Column Names
print("\nDebugging Column Names:")
print(df_predictions.columns)

# Standardizing Column Names
if "Sentiment" in df_predictions.columns:
    df_predictions = df_predictions.rename(columns={"Sentiment": "Sentiment_predicted"})

# Sentiment Distribution
plt.figure(figsize=(8, 5))
if "Sentiment_predicted" in df_predictions.columns:
    sns.countplot(data=df_predictions, x="Sentiment_predicted", hue="Sentiment_predicted", palette="coolwarm", legend=False)
    plt.title("Sentiment Distribution in Customer Calls")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Count")
    plt.show()
else:
    print("Column 'Sentiment_predicted' not found.")

# Resolution Rate (Issue Resolved vs. Follow-up Needed)
if "Call Outcome" in df_predictions.columns:
    resolution_counts = df_predictions["Call Outcome"].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 4))
    sns.barplot(x=resolution_counts.index, y=resolution_counts.values, hue=resolution_counts.index, palette="coolwarm", legend=False)
    plt.title("Resolution Rate: Issue Resolved vs. Follow-up Needed")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Call Outcome")
    plt.ylim(0, 100)
    plt.show()
else:
    print("Column 'Call Outcome' not found.")

# Sentiment vs. Call Outcome Heatmap ---
if "Call Outcome" in df_predictions.columns and "Sentiment_predicted" in df_predictions.columns:
    plt.figure(figsize=(8, 6))
    sentiment_vs_resolution = pd.crosstab(df_predictions["Sentiment_predicted"], df_predictions["Call Outcome"], normalize="index") * 100
    sns.heatmap(sentiment_vs_resolution, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title("Sentiment vs. Call Outcome")
    plt.xlabel("Call Outcome")
    plt.ylabel("Predicted Sentiment")
    plt.show()
else:
    print("Missing 'Sentiment_predicted' or 'Call Outcome'.")

# Word Cloud of Negative & Positive Sentiments
if "Customer Statement" in df_predictions.columns and "Sentiment_predicted" in df_predictions.columns:
    negative_text = " ".join(df_predictions[df_predictions["Sentiment_predicted"] == "Negative"]["Customer Statement"])
    positive_text = " ".join(df_predictions[df_predictions["Sentiment_predicted"] == "Positive"]["Customer Statement"])

    # Negative Sentiment Word Cloud
    plt.figure(figsize=(8, 5))
    wordcloud_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)
    plt.imshow(wordcloud_neg, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Negative Sentiments")
    plt.show()

    # Positive Sentiment Word Cloud
    plt.figure(figsize=(8, 5))
    wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)
    plt.imshow(wordcloud_pos, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Positive Sentiments")
    plt.show()
else:
    print("Missing 'Customer Statement' or 'Sentiment_predicted'.")

# Top 10 Recurring Customer Complaints
if "Customer Statement" in df_predictions.columns and "Sentiment_predicted" in df_predictions.columns:
    negative_statements = df_predictions[df_predictions["Sentiment_predicted"] == "Negative"]["Customer Statement"]
    words = " ".join(negative_statements).split()
    common_words = Counter(words).most_common(10)

    # Plot Top 10 Complaints
    plt.figure(figsize=(8, 5))
    words, counts = zip(*common_words)
    sns.barplot(x=list(counts), y=list(words), hue=list(words), palette="Reds", legend=False)
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Top 10 Recurring Customer Complaints")
    plt.show()
else:
    print("Missing 'Customer Statement'.")

print("\n **Comprehensive insights generation completed!**")
