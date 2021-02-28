import pickle
from hazm import Stemmer, stopwords_list
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np 
import re

stemmer = Stemmer()
stopwords = stopwords_list()

# load models
with open('models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/count_vect.pkl', 'rb') as f:
    count_vect = pickle.load(f)

def pipeline(inp):
    stemmed_input = [stemmer.stem(re.sub("/[a-zA-Z0-9]+/", '',x)) for x in inp.split(' ') if x not in stopwords]
    stemmed_input = ' '.join(stemmed_input)
    cnt = count_vect.transform([stemmed_input])
    tf = tfidf.transform(cnt)
    res = clf.predict_proba(tf)
    return res

with open('products/products.txt', 'r') as f:
    prods = f.readlines()

# predict probability vector for each product
labels = [pipeline(prod)[0] for prod in prods]

# save to disk
np.save('products/product_labels.npy', labels)