from sklearn.externals import joblib
import re
from stop_words import get_stop_words
from nltk.stem.snowball import RussianStemmer

stemmer = RussianStemmer(ignore_stopwords=True)
import numpy as np

N = 750

all_themes = {1:'science', 2:'style', 3:'culture', 4:'life',
              5:'economics', 6:'business', 7:'travel', 8:'forces', 9:'media', 10:'sport'}


clf = joblib.load('clf.pkl')

top = []
with open('news/frequencies.txt', 'r') as freqs:
    print('freqs')
    i = 0
    for word in freqs:
        word = word.strip('\n')
        i += 1
        if i % 100 == 0:
            print(i)
        top.append(word)

#print(top)



with open('news/results.txt', 'w') as results, \
        open('news/news_test.txt', 'r') as test:
        #open('news/test-test.txt', 'r') as test:

    i = 0
    for line in test:
        if i % 100 == 0:
            print(i)
        i += 1
        title, article = line.split('\t')
        text = (title + ' ' + article).strip().lower()
        words = re.split(r'[\s,«»":().-]+', text)
        words = [stemmer.stem(t) for t in words if (t not in set(get_stop_words('ru'))) and (len(t) >= 4)]
        s = set(words)
        y = np.zeros(N)
        j=-1
        for th in top:
            j+=1
            if th in s: y[j] = 1
        #print(y)
        ans = (clf.predict(y))[0]
        #print(ans)
        results.write(all_themes[ans] + '\n')

