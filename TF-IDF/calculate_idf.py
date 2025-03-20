import json
from collections import defaultdict
import math
import pandas as pd

def tokenize(text):
    words = text.split(" ")
    return [word.lower() for word in words if word.isalnum()]

def load_stopwords(stopwords_path):
    with open(stopwords_path, "r") as f:
        return {line.strip() for line in f}

def inverse_document_frequencies(documents):
    num_documents = len(documents)
    idf = defaultdict(int)
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            idf[word] += 1
    return {word: math.log(num_documents / count) for word, count in idf.items()}


data_path = "data.csv" 
stopwords_path = "stopwords-en.txt"
data = pd.read_csv(data_path)

stop_words = load_stopwords(stopwords_path) | {"joe", "biden"}
documents = [
    [word for word in tokenize(title) if word not in stop_words] 
    for title in data['title']
]

idf_values = inverse_document_frequencies(documents)

idf_output_path = "idf_values.json"
with open(idf_output_path, "w") as f:
    json.dump(idf_values, f)

print(f"IDF values saved to {idf_output_path}")
