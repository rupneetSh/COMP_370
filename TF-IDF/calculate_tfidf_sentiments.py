import json
from collections import Counter
import pandas as pd

def tokenize(text):
    words = text.split(" ")
    return [word.lower() for word in words if word.isalnum()]

def term_frequencies(tokens):
    return Counter(tokens)

def compute_tfidf(tf, idf):
    return {word: tf[word] * idf.get(word, 0) for word in tf}

def load_stopwords(stopwords_path):
    with open(stopwords_path, "r") as f:
        return {line.strip() for line in f}

def calculate_tfidf_by_sentiment(data, idf_values, stop_words):
    sentiments = data['Sentiment'].unique()
    sentiment_texts = {sentiment: [] for sentiment in sentiments}
    
    for _, row in data.iterrows():
        sentiment = row['Sentiment']
        tokens = tokenize(row['title'])
        filtered_tokens = [token for token in tokens if token not in stop_words]
        sentiment_texts[sentiment].append(filtered_tokens)
 
    sentiment_tfidf = {}
    for sentiment, documents in sentiment_texts.items():
        tf = Counter([token for tokens in documents for token in tokens])
        tfidf = compute_tfidf(tf, idf_values)
        top_words = sorted(tfidf.items(), key=lambda x: -x[1])[:10]
        sentiment_tfidf[sentiment] = top_words
    
    return sentiment_tfidf


data_path = "data.csv"  
idf_path = "idf_values.json"  
stopwords_path = "stopwords-en.txt"
data = pd.read_csv(data_path)

stop_words = load_stopwords(stopwords_path) | {"joe", "biden"}
with open(idf_path, "r") as f:
    idf_values = json.load(f)

tfidf_results = calculate_tfidf_by_sentiment(data, idf_values, stop_words)

output_path = "tfidf_sentiments.json"
with open(output_path, "w") as f:
    json.dump(tfidf_results, f)

print(f"TF-IDF results for sentiments saved to {output_path}")
