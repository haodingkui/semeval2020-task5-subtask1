# SVM baseline for SemEval-2020 Subtask-1

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

np.random.seed(500)

print(">> Read data...")
# path = $data_path$
path = 'data/train.csv'
corpus = pd.read_csv(path, encoding='utf-8')
percent = 0.3 	# 0.3 for testing
print("File: %s" % path)

corpus['sentence'].dropna(inplace=True)
corpus['sentence'] = [sent.lower() for sent in corpus['sentence']]
corpus['sentence'] = [word_tokenize(word) for word in corpus['sentence']]
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index, entry in enumerate(corpus['sentence']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    corpus.loc[index, 'sentence_final'] = str(Final_words)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['sentence_final'], corpus['gold_label'], test_size=percent)

print(">> Feature generation...")
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus['sentence_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(">> SVM classifier....")
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(Test_Y, predictions_SVM) * 100)
print("SVM Precision Score -> ", precision_score(Test_Y, predictions_SVM) * 100)
print("SVM Recall Score -> ", recall_score(Test_Y, predictions_SVM) * 100)
print("SVM F1 Score -> ", f1_score(Test_Y, predictions_SVM) * 100)

# >> Read data...
# File: data/train.csv
# >> Feature generation...
# >> SVM classifier....
# SVM Accuracy Score ->  88.84615384615384
# SVM Precision Score ->  73.46938775510205
# SVM Recall Score ->  7.860262008733625
# SVM F1 Score ->  14.201183431952662
