import pandas as pd

file_path = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\balanced_final_classified_results.csv"
df = pd.read_csv(file_path)

print(df["Sentiment"].value_counts())
