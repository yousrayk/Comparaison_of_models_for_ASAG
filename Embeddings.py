import torch
from nltk.tokenize import word_tokenize
from allennlp.commands.elmo import ElmoEmbedder
#allennlp.commands.elmo doesn't exists anymore #4384 only v0.90
from bert_embedding import BertEmbedding
from transformers import AutoTokenizer, GPT2LMHeadModel
class WordEmbedder:
    def __init__(self):
        pass

    def elmo_embedding(self, sentence):
        
        embeddings_instance = self.Embedding(sentence)
        elmo_embedding_result = embeddings_instance.get_elmo_embedding()
        word_array = [elmo_embedding_result[0][i] for i in range(len(elmo_embedding_result[2]))]
        return word_array

    def bert_embedding(self, sentence):
        embeddings_instance = self.Embedding(sentence)
        bert_embedding_result = embeddings_instance.get_bert_embedding()
        word_array = [bert_embedding_result[i][1][0] for i in range(len(bert_embedding_result))]
        return word_array

    def gpt2_embedding(self, sentence):

        embeddings_instance = self.Embedding(sentence)
        gpt2_embedding_result = embeddings_instance.get_gpt2_embedding()
        word_array = [gpt2_embedding_result[0][0][i].tolist() for i in range(gpt2_embedding_result[0].size()[1])]
        return word_array

    class Embedding:
        def __init__(self, sentence):
            self.tokenized_sentence = word_tokenize(sentence) if not isinstance(sentence, list) else sentence

        def get_elmo_embedding(self):
            #create a pretrained elmo model (requires internet connection)
            elmo = ElmoEmbedder()
            elmo_embedding_result = elmo.embed_sentence(self.tokenized_sentence)
            return elmo_embedding_result

        def get_bert_embedding(self):
            bert_embedding_result = BertEmbedding().embedding(sentences=self.tokenized_sentence)
            return bert_embedding_result

        def get_gpt2_embedding(self):
            tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
            model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'gpt2')
            indexed_tokens = tokenizer.convert_tokens_to_ids(self.tokenized_sentence)
            tokens_tensor = torch.tensor([indexed_tokens])
            gpt2_embedding_result = model(tokens_tensor)
            return gpt2_embedding_result
'''
Class WordEmbedder:
This class is designed for embedding words or tokens using various language models such as ELMo,
BERT, and GPT-2. It has methods (elmo_embedding, bert_embedding, gpt2_embedding) that take a
sentence as input and return a list of word embeddings.

Class Embeddings:
This is a nested class within WordEmbedder responsible for obtaining embeddings from different
language models. It initializes with a sentence, tokenizes it using NLTK if it's not already a
list of tokens. Methods (get_elmo_embedding, get_bert_embedding, get_gpt2_embedding) return embeddings
from ELMo, BERT, and GPT-2, respectively.

Usage:
Create an instance of WordEmbedder.
Call methods like elmo_embedding, bert_embedding, etc., passing a sentence as an
argument to get word embeddings.
'''