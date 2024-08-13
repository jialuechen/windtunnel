import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze_text(self, text):
        vader_score = self.sia.polarity_scores(text)['compound']
        textblob_score = TextBlob(text).sentiment.polarity
        finbert_result = self.finbert(text)[0]
        finbert_score = finbert_result['score'] if finbert_result['label'] == 'positive' else -finbert_result['score']
        
        return (vader_score + textblob_score + finbert_score) / 3

class MarketSentimentAnalyzer:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def analyze_market_sentiment(self, news_data):
        sentiments = news_data['text'].apply(self.sentiment_analyzer.analyze_text)
        return sentiments.mean()

class SentimentAdjustedOptimizer:
    def __init__(self, base_optimizer, market_sentiment_analyzer, sentiment_adjustment_factor=0.1):
        self.base_optimizer = base_optimizer
        self.market_sentiment_analyzer = market_sentiment_analyzer
        self.sentiment_adjustment_factor = sentiment_adjustment_factor

    def optimize(self, universe, news_data):
        base_weights = self.base_optimizer.optimize(universe)
        market_sentiment = self.market_sentiment_analyzer.analyze_market_sentiment(news_data)
        
        adjusted_weights = base_weights * (1 + self.sentiment_adjustment_factor * market_sentiment)
        return adjusted_weights / adjusted_weights.sum()