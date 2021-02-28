import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
from hazm import Stemmer, stopwords_list
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def get_literal(x):
    lit = literal_eval(x)
    subs = []
    for key in lit.keys():
        subs.append((key,list(lit[key]['sub_category'].keys())))
    return subs

def try_join(x):
    try:
        l = [stemmer.stem(w) for w in literal_eval(x) if w not in stopwords]
        return ' '.join(l)
    except:
        return np.nan

# farsi stemmer and stopwords from hazm
stemmer = Stemmer()
stopwords = stopwords_list()

print("Processing words...")

# remove stopwords and stem words in wikipedia corpus

with open('datasets/wiki.txt', 'r') as f:
    wiki = f.readlines()
words = [w.split(' ') for w in wiki]
words = [item for sublist in words for item in sublist]
words = np.unique(words)
words = np.fromiter((stemmer.stem(xi) for xi in words if xi not in stopwords), words.dtype)

# fit count vectorizer on wikipedia corpus
count_vect = CountVectorizer(ngram_range=(1,2))
count_vect.fit(words)

print("Processing documents...")

web = pd.read_csv('datasets/web.csv', header=None)

web = web.drop(columns=[0,1,2,3,4,5,6,8,12,13,11])

# create bag of words for each entry, cleaning from stopwords and stemming
web['bag_of_words'] = web[7].apply(try_join)
web = web[~web['bag_of_words'].isna()]

# oversampling to eliminate class imbalance
max_size = web[9].value_counts().max()
lst = [web]
for class_index, group in web.groupby(9):
    lst.append(group.sample(max_size-len(group), replace=True))
web = pd.concat(lst)

# shuffle
web = web.sample(frac=1)

# transform bags of words
X_counts = count_vect.transform(web['bag_of_words'])

# tfidf vectorizer
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)

from sklearn.linear_model import SGDClassifier

print("Training classifier...")

# train SGD on train set
clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter_no_change=5)
clf.fit(X_tfidf[:-1000, :],web[9].values[:-1000])

# evaluate on train and test set
train_accuracy = np.mean( clf.predict(X_tfidf[:-1000, :])==web[9].values[:-1000])
test_accuracy = np.mean( clf.predict(X_tfidf[-1000:, :])==web[9].values[-1000:])
random_accuracy = np.mean( np.random.choice(clf.classes_, 1000)==web[9].values[-1000:])

print("Train accuracy: "+str(train_accuracy)[:6])
print("Test accuracy: "+str(test_accuracy)[:6])
print("Compared to random baseline: "+str(random_accuracy)[:6])

# save models
import pickle
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('models/count_vect.pkl', 'wb') as f:
    pickle.dump(count_vect, f)