import re
from stop_words import get_stop_words
from nltk.stem.snowball import RussianStemmer

freq = {}
stemmer = RussianStemmer(ignore_stopwords=True)

N = 750



#with open('news/test-in.txt', 'r') as fin, \
with open('news/news_train.txt', 'r') as fin, \
        open('news/themes.txt', 'w') as themes, \
        open('news/frequencies.txt', 'w') as freqs:
    i = 0
    for line in fin:
        if i % 100 == 0:
            print(i)
        i += 1

        theme, title, article = line.split('\t')
        themes.write(theme + '\n')

        text = (title + ' ' + article).strip().lower()
        words = re.split(r'[\s,«»":().-]+', text)
        words = [stemmer.stem(t) for t in words if (t not in set(get_stop_words('ru'))) and (len(t) >= 4)]
        for t in words:
            if t in freq:
                freq[t] += 1
            else:
                freq.update({t: 1})


    s = [(k, freq[k]) for k in sorted(freq, key=freq.get, reverse=True)]
    i = 0;
    for k, v in s:
        freqs.write(k +'\n')
        i+=1
        if i == N: break