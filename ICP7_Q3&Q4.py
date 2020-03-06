from bs4 import BeautifulSoup
import urllib.request


def search_spider():

    url = "https://en.wikipedia.org/wiki/Google"
    source_code = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_code, "html.parser")

    body = soup.find('div', {'class': 'mw-parser-output'})
    file.write(str(body.text))


search = input('type "s" to start wikiScrap, type "q" to exit')
if search == 'q' or search == 'Q':
    print("Quiting...")
    exit()
else:
    print("Creating .txt file ...")
    file = open('input.txt', 'a+', encoding='utf-8')
    search_spider()
    
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = open('input.txt', encoding="utf8").read()

# a.Tokenization
stokens = nltk.sent_tokenize(sentence)
wtokens = nltk.word_tokenize(sentence)

print(wtokens)
print(stokens)

#c. Stemming
from nltk.stem import PorterStemmer,LancasterStemmer,SnowballStemmer

pStemmer = PorterStemmer()
lStemmer = LancasterStemmer()
sStemmer = SnowballStemmer('english')

n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 4:
        print("Using Porter stemer, Lancaster, Snowballs for"+" "+t+":"+pStemmer.stem(t), lStemmer.stem(t), sStemmer.stem(t))

# b.POS
# d.Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 6:
        print("Lemmatizer:", lemmatizer.lemmatize(t), ",    With POS=a:", lemmatizer.lemmatize(t, pos="a"))


# Trigram
from nltk.util import ngrams

n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)


# Named Entity Recognition
from nltk import word_tokenize, pos_tag, ne_chunk
n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(word_tokenize(s))))
        
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
tfidf_Vect1 = TfidfVectorizer(ngram_range=(1, 2))
tfidf_Vect2 = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
X_train_tfidf1 = tfidf_Vect1.fit_transform(twenty_train.data)
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train.data)


clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

clf1 = MultinomialNB()
clf1.fit(X_train_tfidf1, twenty_train.target)

clf2 = MultinomialNB()
clf2.fit(X_train_tfidf2, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = round(metrics.accuracy_score(twenty_test.target, predicted), 4)
print("MultinomialNB accuracy is: ", score)


twenty_test1 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf1 = tfidf_Vect1.transform(twenty_test.data)

predicted1 = clf1.predict(X_test_tfidf1)

score1 = round(metrics.accuracy_score(twenty_test1.target, predicted1), 4)
print("MultinomialNB accuracy when using bigram is: ", score1)

twenty_test2 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf2 = tfidf_Vect2.transform(twenty_test.data)

predicted2 = clf2.predict(X_test_tfidf2)

score2 = round(metrics.accuracy_score(twenty_test2.target, predicted2), 4)
print("MultinomialNB accuracy when adding the stop-words is: ", score2)

        
#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

svc = KNeighborsClassifier(n_neighbors=2)
svc.fit(X_train_tfidf, twenty_train.target)

acc_knn = round(svc.score(X_train_tfidf, twenty_train.target) * 100, 2)
print("SVC accuracy is:", acc_knn / 100)