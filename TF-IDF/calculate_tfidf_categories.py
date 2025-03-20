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

def calculate_tfidf_by_category(data, idf_values, stop_words):
    categories = data['Category'].unique()
    category_texts = {category: [] for category in categories}
    
    for _, row in data.iterrows():
        category = row['Category']
        tokens = tokenize(row['title'])
        filtered_tokens = [token for token in tokens if token not in stop_words]
        category_texts[category].append(filtered_tokens)
    
    category_tfidf = {}
    for category, documents in category_texts.items():
        tf = Counter([token for tokens in documents for token in tokens])
        tfidf = compute_tfidf(tf, idf_values)
        top_words = sorted(tfidf.items(), key=lambda x: -x[1])[:10]
        category_tfidf[category] = top_words
    
    return category_tfidf

data_path = "data.csv"
idf_path = "idf_values.json" 
stopwords_path = "stopwords-en.txt"
data = pd.read_csv(data_path)

stop_words = load_stopwords(stopwords_path) | {"joe", "biden"}
with open(idf_path, "r") as f:
    idf_values = json.load(f)

tfidf_results = calculate_tfidf_by_category(data, idf_values, stop_words)

output_path = "tfidf_categories.json"
with open(output_path, "w") as f:
    json.dump(tfidf_results, f)

print(f"TF-IDF results for categories saved to {output_path}")
