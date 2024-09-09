from transformers import pipeline
import pandas as pd

data = pd.read_csv("./data/trasnformed_output.csv")

def feature_extraction():
    classifier = pipeline("feature-extraction")

def ner():
    classifier = pipeline("ner")

def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")

