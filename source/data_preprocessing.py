import re
from functools import reduce

from nltk import word_tokenize
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))
repls = ("(", ""), (")", ""), ("'s", "")


def tokenize(text):
    lev1 = re.sub("[!#*%:,.;&-]", "", text)                         # Remove specific chars
    lev2 = re.sub(r'[^\x00-\x7f]', r' ', lev1)                      # Remove non ASCII chars
    lev3 = reduce(lambda a, kv: a.replace(*kv), repls, lev2)        # Replace using functional approach
    tokens = map(lambda word: word.lower(), word_tokenize(lev3))    # Lowercase strings
    words = [word for word in tokens if word not in stop]           # Select words not present in stopwords set
    return words


def join_strings(x):
    return " ".join(sorted(x)).strip()
