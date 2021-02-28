import pickle
from hazm import Stemmer, stopwords_list
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np 
import sys

stemmer = Stemmer()
stopwords = stopwords_list()

try:
    n_top = int(sys.argv[1])
except:
    n_top = 5

# load models
with open('models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/count_vect.pkl', 'rb') as f:
    count_vect = pickle.load(f)

def pipeline(inp):
    stemmed_input = [stemmer.stem(x) for x in inp.split(' ') if x not in stopwords]
    stemmed_input = ' '.join(stemmed_input)
    cnt = count_vect.transform([stemmed_input])
    tf = tfidf.transform(cnt)
    res = clf.predict_proba(tf)
    return res

with open('input.txt', 'r') as f:
    inputs = f.readlines()

with open('products/products.txt', 'r') as f:
    prods = f.readlines()
prods = np.array(prods)

# load predicted vectors for products
prod_labels = np.load('products/product_labels.npy')

with open('results.txt', 'w') as f:
    for inp in inputs:
        inp_probs = pipeline(inp)
        
        # get similarity between input and products
        sims = np.array([np.dot(inp_probs, prod) for prod in prod_labels]).flatten()
        
        # get top n products
        prod_args = np.argsort(-sims)[:n_top]
        print(inp,file=f)
        for prod_arg in prod_args:
            print(prods[prod_arg],file=f)
        print('---------------------------',file=f)