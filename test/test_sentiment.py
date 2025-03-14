import pandas as pd

# Load sentiment results
predictions_file = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\balanced_final_classified_results.csv"
df = pd.read_csv(predictions_file)

def test_sentiment():
    valid_labels = {"Positive", "Negative", "Neutral"}
    
    assert "Sentiment" in df.columns, "Sentiment column missing!"
    assert df["Sentiment"].isin(valid_labels).all(), "Invalid sentiment labels detected!"
    
    print("Sentiment classification test passed!")

if __name__ == "__main__":
    test_sentiment()
