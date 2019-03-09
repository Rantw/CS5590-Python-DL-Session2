# Ronnie Antwiler
# 1.Extract the following web URL text using BeautifulSoup
# https://en.wikipedia.org/wiki/Google
# 2. Save it in input.txt

# 3. Apply the following on the text and show output:
# a. Tokenization
# b. POS
# c. Stemming
# d. Lemmatization
# e. Trigram
# f. Named Entity Recognition

import requests
from bs4 import BeautifulSoup

# Part 1
# Get Wiki information
url = 'https://en.wikipedia.org/wiki/Google'
source = requests.get(url)
plain = source.text
soup = BeautifulSoup(plain, 'html.parser')

# Part 2
# Write to text file
file2 = open('input'+'.txt','a+',encoding='utf-8')
body = soup.find('div', {'class': 'mw-parser-output'})
file2.write(str(body.text))

# Part 3
# I limited printed samples to 5
# Part 3a Tokenization
import nltk
#nltk.download()

# Get text from text file
with open('input.txt', encoding='utf8') as data:
    text = data.read().strip()

# Tokenize the words and sentences
stokens = nltk.sent_tokenize(text)
wtokens = nltk.word_tokenize(text)

# Print out a sample of words tokens
sample = 0
for s in wtokens:
    print(s)
    sample += 1
    if sample >= 5:
        break
print()
print()

# Print out a sample of sentence tokens
sample1 = 0
for s in stokens:
    print(s)
    sample1 += 1
    if sample1 >= 5:
        break
print()
print()

# Part 3b Part of Speech tagging
textpos = nltk.pos_tag(wtokens)

print('POS')
sample2 = 0
for s in textpos:
    print(s)
    sample2 += 1
    if sample2 >= 5:
        break
print()
print()

# Part 3c Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

pStem = PorterStemmer()
lStem = LancasterStemmer()
sStem = SnowballStemmer('english')

print('Porter Stemmer')
sample3 = 0
for s in wtokens:
    print(pStem.stem(s))
    sample3 += 1
    if sample3 >= 5:
        break
print()
print()
print('Lancaster Stemmer')
sample4 = 0
for s in wtokens:
    print(lStem.stem(s))
    sample4 += 1
    if sample4 >= 5:
        break
print()
print()
print('Snowball Stemmer')
sample5 = 0
for s in wtokens:
    print(sStem.stem(s))
    sample5 += 1
    if sample5 >= 5:
        break
print()
print()
# Part 3d Lemmatization
from nltk.stem import WordNetLemmatizer

lemmat = WordNetLemmatizer()

print('Lemmatizer')
sample6 = 0
for s in wtokens:
    print(lemmat.lemmatize(s))
    sample6 += 1
    if sample6 >= 5:
        break
print()
print()
# Part 3e. Trigram
from nltk.util import ngrams
trigrams = ngrams(wtokens,3)

print('Trigrams')
sample8 = 0
for s in trigrams:
    print(s)
    sample8 += 1
    if sample8 >= 5:
        break

print()
print()

# Part 3f. Named Entity Recognition
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

sample7 = 0
print('Named Entity Recognition')
for s in stokens:
    print(ne_chunk(pos_tag(wordpunct_tokenize(s))))
    sample7 += 1
    if sample7 >= 5:
        break
print()
print()