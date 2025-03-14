**AXA Gen AI - Sentiment Analysis and Insights**

**Project Overview**

This project is a sentiment analysis and insights pipeline for customer transcripts using fine-tuned NLP models. It consists of:

+ **Sentiment Classification**: Fine-tuning transformer models on customer sentiment data.

+ **Ground Truth Labeling**: Generating high-quality sentiment labels for evaluation.

+ **Model Evaluation**: Comparing predictions with ground truth data using various performance metrics.

+ **Insights & Analytics**: Extracting valuable business insights from the processed data.

**Project Structure**
```bash
Axa_Gen_AI/
│── datasets/                                    # Raw and processed datasets
│   ├── Transcripts_v3 - Dummy Data/             # Example customer transcripts
│── fine_tuned_sentiment_model/                  # Checkpoints and model files
│── presentations/                               # Reports & visualizations
│── results/                                     # Processed results and evaluation outputs
│   ├── final_classified_results.csv             # Classified transcripts
│   ├── ground_truth_dataset.csv                 # Generated ground truth labels
│   ├── *.png                                    # Visualization outputs (confusion matrix, etc.)
│   ├── *.csv                                    # Classified results, ground truth data
│── src/                                         # Source code for processing and analysis
│   ├── check_miss_match_ground_final.py         # Mismatch detection
│   ├── fine_tune_sentiment.py                   # Fine-tune sentiment analysis model
│   ├── generating_groundtruth_label_dataset.py  # Generate ground truth labels
│   ├── insights.py                              # Exploratory data analysis
│   ├── model_evaluation.py                      # Model evaluation script
│   ├── preprocess.py                            # Preprocessing pipeline
│   ├── resample.py                              # Data resampling techniques
│   ├── sentiment_analysis.py                    # Sentiment classification script
│── test/                                        # Unit test scripts
│   ├── test_preprocessing.py
│   ├── test_sentiment.py
│── requirements.txt                             # Python dependencies
│── README.md                                    # Project documentation


Setup and Installation

1)  Create & Activate a Virtual Environment
    conda create --name axa_gen_ai python=3.8 -y
    conda activate axa_gen_ai

2️) Install Dependencies
   pip install -r requirements.txt

Running the Pipeline

1)  Preprocessing Customer Statements
    python src/preprocess_1.py

2)  Running Sentiment Analysis
    python src/sentiment_analysis_2.py

3)  Resampling
    resample_3.py

4)  Fine-Tuning the Sentiment Model
    python src/fine_tune_sentiment_4.py

5)  Generating Ground Truth Sentiment Labels
    python src/generating_groundtruth_label_dataset_5.py

6)  Evaluating Model Performance
    python src/model_evaluation_6.py

6)  Generating Business Insights
    python src/insights_.py 

```

**Key Features & Insights**

+ Confusion Matrix: Evaluates misclassifications

+ Sentiment Distribution: Analyzes trends across transcripts

+ Resolution Rate: Determines issue resolution efficiency

+ Word Cloud Analysis: Highlights frequent phrases in positive & negative reviews

+ Call Length vs. Resolution: Identifies trends in handling times

**Future Enhancements**

+ Deploy as an API for real-time sentiment classification

+ Integrate with customer support dashboards

+ Explore multi-lingual sentiment analysis


