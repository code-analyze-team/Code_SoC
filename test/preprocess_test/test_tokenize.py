from build_data.preprocess import preprocessor
from nltk.corpus import wordnet
import enchant
import collections

input = 'orgmybatisspringsampledbdatabasetestdata ibati properti'

res = preprocessor.tokenize(input)
res = preprocessor.remove_stop_words(res)
res = preprocessor.lemmatize(res)
res = preprocessor.stemm(res)
print(res)