# NLP Product Recommender
### Alireza Hashemi

## Discussion

We are going to desing a system which recommends products from Digikala based on an input string.
In order to do this, we will train a document classifier model on the provided dataset, using the wikipedia crawled words as a corpus. We will be using Hazm Farsi stemmer and stopwords to improve performance of the classification.

In order to eliminate the class imbalance in web.csv dataset, we use oversampling. We create bags of words for each entry and then we use CountVectorizer with ngram_range of (1,2). This hyperparameter and the one which follows are results of an small GridSearch.
We use SGDClassifier, it has proven to work better the Naive Bayes on this dataset. Although using hinge loss results in an slightly better performance, in order to be able to retrieve probabilities for each class, we use log loss.
This model reaches ~0.81 accuracy on test set.

Now we can label the Digikala products with this model, I save the model prediction as a vector with n_classes numbers in it.
We do the same for the input string then the similarity of the input and each product would be defined as their cosine similarity, namely the dot product of their probability vectors. Using this, we can retrieve top recommendations for an input string.

### What would I improve if I had more time?

I would train document classifier on subclasses in web.csv too, it probably would results in a better separation between topics.
I would also try training ANNs as document classifier.

-------------------------------------------------------------------------------

## How to run?

### 1) Crawling Digikala

Running 
python3 digikala_crawler.py 1 

will crawl Digikala starting URLs for products, each url will be crawled for 1 page. You can increase this factor for more products.
This will generate a products.txt file in products folder.

### 2) Training classifiers
If you want to use my trained models, skip this and 3 and go to 4
Now we need to place the two files, web.csv & wiki.txt in the datasets folder. Now we can train our model.
python3 train_classifier.py

This will report train/test accuracy and generate pickle files for models.

### 3) Labeling products
If you want to use my trained models, skip this and go to 4
Now we need to label the products crawled from Digikala.
python3 classify_products.py

This will create a product_labels.npy in the products folder.

### 4) Predicting on input

Enter your string input in input.txt, then
python3 predict.py 5

will generate 5 product recommendations into result.txt for each line in input.txt.

