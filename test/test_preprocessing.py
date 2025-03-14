import json
import os

# Load processed data
processed_file = r"C:\Uday\Constant\AXA Insurance\Axa_Gen_AI\results\processed_transcripts.json"

def test_preprocessing():
    assert os.path.exists(processed_file), "Processed data file not found!"
    
    with open(processed_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    assert isinstance(data, dict), "Processed data should be a dictionary!"
    
    for key, statements in data.items():
        assert isinstance(statements, list), f"Customer statements in {key} should be a list!"
        assert all(isinstance(s, str) for s in statements), f"All statements should be strings in {key}!"
        assert all(len(s) > 0 for s in statements), f"Empty statement found in {key}!"
    
    print("Preprocessing test passed!")

if __name__ == "__main__":
    test_preprocessing()
