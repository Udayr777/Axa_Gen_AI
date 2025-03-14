import os
import re
import json

# Path to dataset directory
DATASET_PATH = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\datasets\Transcripts_v3 - Dummy Data\transcripts_v3"

# Function to clean customer statements which remove noise
def clean_text(text):
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", text)  # Removes timestamps
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)  # Keep meaningful characters
    text = text.strip()  # Remove extra spaces
    return text

# Function to extract customer (Member) statements from transcript files
def extract_customer_statements(file_path):
    customer_lines = []    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # Match variations of "Member"
                if re.match(r"(?i)^\s*Member\s*:", line):
                    clean_line = re.sub(r"(?i)^\s*Member\s*:", "", line).strip()  # Remove "Member" prefix
                    cleaned_text = clean_text(clean_line)  # Clean extracted text
                    if cleaned_text:  # Only add non-empty statements
                        customer_lines.append(cleaned_text)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return customer_lines

# Function to process all transcripts and save structured output
def process_transcripts():
    transcripts_data = {}

    for transcript_file in os.listdir(DATASET_PATH):
        file_path = os.path.join(DATASET_PATH, transcript_file)

        # Extract customer statements
        customer_statements = extract_customer_statements(file_path)

        # Only add files that have valid customer statements
        if customer_statements:
            transcripts_data[transcript_file] = customer_statements
        else:
            print(f"Skipping empty transcript: {transcript_file}")

    # Save processed data to JSON format
    output_path = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\processed_transcripts.json"
    
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(transcripts_data, json_file, indent=4)

    print(f"Processed data saved to: {output_path}")

# Run the script
if __name__ == "__main__":
    process_transcripts()
