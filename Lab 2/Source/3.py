import nltk
import collections
import numpy
from nltk.stem  import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams,ne_chunk,wordpunct_tokenize,pos_tag,FreqDist
with open('nlp.txt', 'r') as f:
  raw=f.read()
#Tokenization
wtokens=nltk.word_tokenize(raw)
print(wtokens)
lemmatizer = WordNetLemmatizer()
print("\nLemmatization ------------------------------------------------------------>>>>>>:\n")
for tok in wtokens:
  print(lemmatizer.lemmatize(str(tok)))
print("\nTrigrams ------------------------------------------------------------------->>>>>>:\n")
trigram = []
x=0
trigram.append(list(ngrams(wtokens, 3)))
print(trigram)

TrigramOutput = []
for big in ngrams(wtokens, 3):
    # Fetching trigrams using 'ngrams' method and Iterating it
    TrigramOutput.append(big)
print(TrigramOutput)
wordFreq = FreqDist(TrigramOutput)
print(wordFreq)
mostCommon = wordFreq.most_common()
print(mostCommon)
mostCommon1 = wordFreq.most_common(10)
print(mostCommon1)
sentTokens = sent_tokenize(raw)
print(sentTokens)
# Creating an Array to append the sentence
concatenatedArray = []
# Iterating the Sentences
for sentence in sentTokens:
    # Iterating the trigrams present
    for a, b ,c in TrigramOutput:
        # Iterating the Top 5 trigrams
        for ((c, m, n ), length) in mostCommon1:
            # Comparing the each with each of the Top 5 trigrams
            if(a, b, c == c, m, n):
                concatenatedArray.append(sentence)
print("\nMax of Concatenated Array : ", max(concatenatedArray))