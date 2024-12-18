import numpy as np
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import urllib.request
import tarfile
from pathlib import Path

def load_dataset(name_url):
    root_url = "https://spamassassin.apache.org/old/publiccorpus/"
    url = root_url + name_url
    tarball_path = Path(f"email/{name_url.split('.')[0]}")  # Directory to extract files
    # Create root email directory if it doesn't exist
    Path("email").mkdir(parents=True, exist_ok=True)
    # Download the tar file if it doesn't exist
    if not tarball_path.is_file():
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as data_tarball:
            data_tarball.extractall(path="email")
    # Return the path to the extracted directory containing emails

# Datasets
easy_ham = ["20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2"]
hard_ham = ["20030228_hard_ham.tar.bz2"]
spam = ["20030228_spam.tar.bz2", "20050311_spam_2.tar.bz2"]

for name_url in easy_ham+hard_ham+spam:
    load_dataset(name_url)

easy_ham_path = [f for f in sorted(Path("email/easy_ham").iterdir()) if len(f.name) > 20] + [f for f in sorted(Path("email/easy_ham_2").iterdir()) if len(f.name) > 20]
hard_ham_path = [f for f in sorted(Path("email/hard_ham").iterdir()) if len(f.name) > 20] 
spam_path = [f for f in sorted(Path("email/spam").iterdir()) if len(f.name) > 20] + [f for f in sorted(Path("email/spam_2").iterdir()) if len(f.name) > 20]

import email
from email import policy

def load_email(file_path):
    with open(file_path, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

easy_ham_emails = [load_email(f) for f in easy_ham_path]
hard_ham_emails = [load_email(f) for f in hard_ham_path]
spam_emails = [load_email(f) for f in spam_path]

print(len(easy_ham_emails))
print(len(hard_ham_emails))
print(len(spam_emails))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from collections import Counter

X = np.array(easy_ham_emails + spam_emails, dtype="object")
Y = np.array([0]*len(easy_ham_emails) + [1]*len(spam_emails))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from bs4 import BeautifulSoup

def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()
def email_to_text(email):
    total_content = ""
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        else:
            try:
                content = part.get_content()
            except:
                content = str(part.get_payload())
            if ctype == "text/plain":
                total_content += content
            else:
                total_content += html_to_text(content)
    return total_content

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "runs", "run", "happiness", "was"]
lemmas = [lemmatizer.lemmatize(word, pos="v") for word in words]  # pos="v" for verbs

print(lemmas)  # Output: ['run', 'run', 'run', 'run', 'happiness', 'be']

from urlextract import URLExtract

def url_extractor(email):
    url_extract = URLExtract()
    urls = url_extract.find_urls(email)
    return urls

print(url_extractor(email_to_text(X_train[195])))
from sklearn.base import TransformerMixin, BaseEstimator
import re

class EmailtoWordCounterTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, strip_headers=True, to_lowercase=True, replace_numbers=True,
                remove_punctuations=True, replace_urls=True, stemming=True):
        self.strip_headers=strip_headers
        self.to_lowercase=to_lowercase
        self.replace_numbers=replace_numbers
        self.replace_urls = replace_urls
        self.remove_punctuations = remove_punctuations
        self.stemming=stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text=email_to_text(email) or ""
            if self.to_lowercase:
                text=text.lower()
            if self.replace_urls:
                if url_extractor is None:
                    raise ValueError("URL extractor is not initialized!")
                urls = url_extractor(text)
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text=re.sub("\d+(?:\.\d*)?(?:[eE][+-]?\d+(?:\.\d*))?", "NUMBER", text)
            if self.remove_punctuations:
                text=re.sub("\W+", ' ', text)
            word_counts = Counter(text.split())
            if self.stemming:
                if lemmatizer is None:
                    raise ValueError("lemmatizer is not initialized!")
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word_counts[lemmatizer.lemmatize(word, pos="v")] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)        
from scipy.sparse import csr_matrix

class WordCountertoVectorsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_words = Counter()
        for words_counter in X:
            for word, count in words_counter.items():
                total_words[word] += min(count, 10)
        most_common = total_words.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word:index+1 for index, (word, count) in enumerate(most_common)}
        return self
    
    def transform(self, X, y=None):
        vector = []
        col_index = []
        row_index = []
        for row, email_word_counter in enumerate(X):
            for word, count in email_word_counter.items():
                row_index.append(row)
                col_index.append(self.vocabulary_.get(word, 0))
                vector.append(count)
        return csr_matrix((vector, (row_index, col_index)), shape=(len(X), self.vocabulary_size+1))
from sklearn.pipeline import Pipeline

preprocessing = Pipeline([
    ("Words_to_Counter", EmailtoWordCounterTransformer()),
    ("WordCounter_To_Vector", WordCountertoVectorsTransformer())
])
X_train_processed = preprocessing.fit_transform(X_train, Y_train)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_processed)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

svm_clf = SVC(random_state=42)
train_predict_svm = cross_val_predict(svm_clf, X_train_tfidf, Y_train, cv=10)
f1_score(Y_train, train_predict_svm)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

X_test_processed = preprocessing.transform(X_test)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_processed)
svm_clf.fit(X_train_tfidf, Y_train)
Y_pred = svm_clf.predict(X_test_tfidf)
print(f"Precision {precision_score(Y_test, Y_pred):.2%}")
print(f"Recall {recall_score(Y_test, Y_pred):.2%}")
conf = confusion_matrix(Y_test, Y_pred)
print(conf)

hard_ham_processed = preprocessing.transform(np.array(hard_ham_emails, dtype="object"))
hard_ham_tfidf = tfidf_transformer.fit_transform(hard_ham_processed)
hard_ham_predict = svm_clf.predict(hard_ham_tfidf)
Y_orig = np.array([0]*len(hard_ham_emails))
conf1 = confusion_matrix(Y_orig, hard_ham_predict)
print(conf1)

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

svc = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100, 1000, 2000],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 1e-3, 1e-2, 0.1, 1],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000, 5000, 10000, 15000, 20000],
    'degree': [2, 3, 4, 5]
}
rnd_search_svc = RandomizedSearchCV(svc, param_grid, n_iter=100, cv=3, scoring='f1', verbose=2, n_jobs=-1)
rnd_search_svc.fit(X_train_tfidf, Y_train)
print("Best parameters:", rnd_search_svc.best_params_)

cv_res = pd.DataFrame(rnd_search_svc.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res.head()
params = rnd_search_svc.best_params_
svm_clf = SVC(**params)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

svm_clf.fit(X_train_tfidf, Y_train)
Y_pred = svm_clf.predict(X_test_tfidf)
print(f"Precision {precision_score(Y_test, Y_pred):.2%}")
print(f"Recall {recall_score(Y_test, Y_pred):.2%}")
conf = confusion_matrix(Y_test, Y_pred)
conf
hard_ham_predict = svm_clf.predict(hard_ham_tfidf)
Y_orig = np.array([0]*len(hard_ham_emails))
conf1 = confusion_matrix(Y_orig, hard_ham_predict)
print(conf1)
import joblib
joblib.dump(svm_clf, "email_classifier_model.pkl")