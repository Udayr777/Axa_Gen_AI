import json
import pandas as pd
from transformers import pipeline

# Load the best available sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # pre-trained sentiment model
sentiment_analyzer = pipeline("text-classification", model=model_name, tokenizer=model_name, device=0)

# Loading processed transcripts
input_file = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\processed_transcripts.json"
output_file = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\final_classified_results.csv"

with open(input_file, "r", encoding="utf-8") as file:
    transcripts_data = json.load(file)

# Define function for sentiment classification
def classify_sentiment(statement):

    # Classifies sentiment as Positive, Neutral, or Negative based on model confidence scores.
    
    try:
        result = sentiment_analyzer(statement, truncation=True, max_length=512, batch_size=16)[0]  # Batch processing
        label = result["label"]

        # Convert model labels to standard format
        if "positive" in label.lower():
            return "Positive"
        elif "negative" in label.lower():
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error processing: {e}")
        return "Neutral"

# Define function to determine call outcome
def determine_call_outcome(statements):

    # Determines if the call issue was resolved based on sentiment distribution.

    positive_count = sum(1 for s in statements if classify_sentiment(s) == "Positive")
    negative_count = sum(1 for s in statements if classify_sentiment(s) == "Negative")

    return "Issue Resolved" if positive_count > negative_count else "Follow-up Needed"

# Process all transcripts
results = []

for transcript_file, customer_statements in transcripts_data.items():
    sentiments = [classify_sentiment(statement) for statement in customer_statements]
    outcome = determine_call_outcome(customer_statements)

    # Store results
    for statement, sentiment in zip(customer_statements, sentiments):
        results.append({
            "Transcript": transcript_file,
            "Customer Statement": statement,
            "Sentiment": sentiment,
            "Call Outcome": outcome
        })

# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
