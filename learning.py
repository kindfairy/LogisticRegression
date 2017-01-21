import numpy as np
import re
from stop_words import get_stop_words
from nltk.stem.snowball import RussianStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

N = 750
num_of_samples = 60000
#num_of_samples = 100

all_themes = {'science': 1, 'style': 2, 'culture': 3, 'life': 4,
              'economics': 5, 'business': 6, 'travel': 7, 'forces': 8, 'media': 9, 'sport': 10}

themes = []
with open('news/themes.txt', 'r') as fin:
    print('themes')
    for th in fin:
        th = th.strip('\n')
        if th in all_themes:
            themes.append(all_themes[th])

print(themes)

top = []
with open('news/frequencies.txt', 'r') as freqs:
    print('freqs')
    i = 0
    for word in freqs:
        word = word.strip('\n')
        top.append(word)

print(top)


matrix = np.zeros((num_of_samples, N))

#with open('news/test-in.txt', 'r') as fin:
with open('news/news_train.txt', 'r') as fin:
    print('fin')
    i = -1
    for line in fin:
        i += 1
        if i % 100 == 0:
            print(i)

        stemmer = RussianStemmer(ignore_stopwords=True)
        theme, title, article = line.split('\t')
        text = (title + ' ' + article).strip().lower()
        words = re.split(r'[\s,«»":().-]+', text)
        words = [stemmer.stem(t) for t in words if (t not in set(get_stop_words('ru'))) and (len(t) >= 4)]
        s = set(words)

        j = -1
        #print(s)
        for l in top:
            j += 1
            if l in s:
                #print(l)
                matrix[i, j] = 1


clf = LogisticRegression()
clf.fit(matrix, themes)

joblib.dump(clf, 'clf.pkl')