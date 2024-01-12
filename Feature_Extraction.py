import scipy
import scipy.spatial
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error

class EmbeddingOperations:
    def __init__(self):
        pass

    def sowe(self, array):
        return sum(array)

    def mowe(self, array):
        return sum(array) / len(array)

    def gpt_sowe(self, array):
        return [sum(i) for i in zip(*array)]
    

class TextSimilarityMetrics:
    def __init__(self, word_array_1, word_array_2):
        self.word_array_1 = word_array_1
        self.word_array_2 = word_array_2

    def get_cosine_similarity(self, u, v):
        return 1 - scipy.spatial.distance.cosine(u, v)

    def __cosine_similarity_matrix(self):
        matrix = np.zeros((len(self.word_array_1), len(self.word_array_2)))

        for i in range(len(self.word_array_1)):
            for j in range(len(self.word_array_2)):
                matrix[i][j] = self.get_cosine_similarity(self.word_array_1[i], self.word_array_2[j])
        return matrix.T

    def plot_similarity_matrix(self, sentence_1, sentence_2, title):
        x_labels, y_labels = word_tokenize(sentence_1), word_tokenize(sentence_2)
        similarity_matrix = self.__cosine_similarity_matrix()
        sns.heatmap(similarity_matrix, vmin=0, vmax=1, xticklabels=x_labels, yticklabels=y_labels, cmap="YlGnBu",
                    annot=True)
        plt.title(title)
        plt.show()

    def get_similar_words(self, sentence_1, sentence_2):
        token_1, token_2 = word_tokenize(sentence_1), word_tokenize(sentence_2)

        similarity_matrix = self.__cosine_similarity_matrix()

        similar_word_dict = {}
        for row in range(len(similarity_matrix[0])):
            min_val = min(similarity_matrix.T[row])
            index = np.where(similarity_matrix.T[row] == min_val)[0]
            similar_word_list = [token_2[i] for i in index]
            similar_word_dict[token_1[row]] = similar_word_list

        print('Similar words in two sentences are:', similar_word_dict)

class EvaluationMetrics:
    def __init__(self, x, y):
        self.x = self.__check_nan(np.asarray(x))
        self.y = self.__check_nan(np.asarray([round(i * 2) / 2 for i in y]))

    def __check_nan(self, array):
        NaNs_index = np.isnan(array)
        array[NaNs_index] = 0
        return array

    def root_mean_squared_error(self):
        for val in self.y:
            if np.isnan(val) or not np.isfinite(val):
                print(val)
        return sqrt(mean_squared_error(self.x, self.y))

    def pearson_correlation(self):
        mean_x, mean_y = np.mean(self.x), np.mean(self.y)
        cov = np.mean((self.x - mean_x) * (self.y - mean_y))
        std_x, std_y = np.std(self.x), np.std(self.y)
        p = cov / (std_x * std_y)
        return float(p)
'''
Class EmbeddingOperations:
Provides methods for basic operations on word embeddings, such as sum and mean.

Class TextSimilarityMetrics:
Calculates text similarity metrics using cosine similarity between word embeddings.
*get_cosine_similarity: Computes cosine similarity between two vectors.
*__cosine_similarity_matrix: Generates a matrix of cosine similarities for words in two sentences.
*plot_similarity_matrix: Plots a heatmap of the cosine similarity matrix.
*get_similar_words: Finds similar words in the second sentence for each word in the first sentence.

Class EvaluationMetrics:
Performs evaluation metrics on two sets of data (x and y).
*__check_nan: Replaces NaN values in an array with 0.
*root_mean_squared_error: Computes the root mean squared error between two sets of values.
*pearson_correlation: Calculates Pearson correlation coefficient.
*spearman_correlation: Computes Spearman rank-order correlation coefficient.'''