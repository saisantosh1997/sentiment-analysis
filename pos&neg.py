import nltk.classify.util
from nltk.classify import NaiveBayesClassifier # we use NaiveBayes to classify the reviews as positive or negative
from nltk.stem import WordNetLemmatizer #
from nltk.tokenize import word_tokenize # 
import pandas as pd

positive_reviews = pd.read_csv('hotel_positive_reviews.csv')
negative_reviews = pd.read_csv('hotel_negative_reviews.csv')

positive_reviews = positive_reviews[:len(negative_reviews)]
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

wordnet_lemmatizer = WordNetLemmatizer()

word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []
def my_tokenizer(s):
    s = s.lower() # Convert into lowercase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    my_dict = dict([(t, True) for t in tokens])
    return my_dict # return dictionary
neg_reviews = []
for reviews in negative_reviews['review']:
    neg_reviews.append((my_tokenizer(reviews), "negative"))
pos_reviews = []
for reviews in positive_reviews['review']:
    pos_reviews.append((my_tokenizer(reviews), "positive"))
train_set = neg_reviews + pos_reviews
classifier = NaiveBayesClassifier.train(train_set)
hotel = pd.read_excel('hotel_review.xlsx')
tag=[]
for review in hotel['review']:
    tokened=my_tokenizer(review)
    tag.append(classifier.classify(tokened))
hotel['tag']=tag
print(hotel)