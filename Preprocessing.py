import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextProcessor:
    def __init__(self):
        pass

    def tokenize_text(self, text):
        tokens = word_tokenize(text)
        return tokens

    def extract_unique_tokens(self, source, target):
        source_tokens = set(self.tokenize_text(source))
        target_tokens = set(self.tokenize_text(target))
        unique_tokens = target_tokens - source_tokens
        return list(unique_tokens)

    def filter_stop_words(self, tokens):
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        return filtered_tokens
  
'''
The provided Python code defines a class named
TextProcessor with three methods: tokenize_text, extract_unique_tokens, and filter_stop_words.
This class is designed for preprocessing text data, particularly for natural language processing
(NLP) tasks. Let's break down each method:
*tokenize_text(self, text):
This method takes one input parameters, text,
and tokenizes it using the word_tokenize function from 
the Natural Language Toolkit (nltk). It returns a tuple of tokenized list.

*extract_unique_tokens(self, source, target):
This method utilizes the tokenization method to obtain tokenized
versions of the question(source) and answer(target). It then creates a new list
(unique_tokens) containing words from the answer that are not present in the question.
The method returns this list of demoted tokens.

*filter_stop_words(self, tokens): This method removes common English stop words
from the input list of demoted tokens.
It uses the NLTK library's stopwords module to obtain a set of stop words and then
filters out those words from the input list. The resulting list, which contains non-stop words,
is returned. 
See Geeks4Geeks'''