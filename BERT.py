import json
import pandas as pd
from Embeddings import *
from Preprocessing import *
from Feature_Extraction import *

def preprocess_text(question, answer):

    text_processor = TextProcessor()
    unique_tokens = text_processor.extract_unique_tokens(question, answer)
    filtered_tokens = text_processor.filter_stop_words(unique_tokens)
    return filtered_tokens

if __name__ == '__main__':
    # Read dataset from CSV
    df = pd.read_csv('mohler_dataset.csv')

    # Get student answers from the dataset
    student_answers = df['student_answer'].to_list()
    similarity_scores = {}

    # For each student answer, get id, question, and desired answer
    for stu_ans in student_answers:
        record = df.loc[df['student_answer'] == stu_ans].iloc[0]
        id, question, desired_answer = record['id'], record['question'], record['desired_answer']

        # Preprocess student answer and desired answer
        pp_desired = preprocess_text(question, desired_answer)
        pp_student = preprocess_text(question, stu_ans)

        # Assign embeddings to desired answer and student answer
        embedder = WordEmbedder()
        word_array_1 = embedder.bert_embedding(pp_desired)
        word_array_2 = embedder.bert_embedding(pp_student)

        # Compare and assign cosine similarity to the answers
        similarity_metrics = TextSimilarityMetrics(word_array_1, word_array_2)
        embedding_operations = EmbeddingOperations()

        text_1_embed = embedding_operations.sowe(word_array_1)
        text_2_embed = embedding_operations.sowe(word_array_2)

        similarity_scores[stu_ans] = similarity_metrics.get_cosine_similarity(text_1_embed, text_2_embed)
        print(similarity_scores)

    # Save similarity scores to JSON
    with open('bert_similarity_scores.json', 'w') as fp:
        json.dump(similarity_scores, fp)

    # Update DataFrame with the similarity scores
    for answer in student_answers:
        df.loc[df['student_answer'] == answer, 'similarity_score'] = similarity_scores[answer]

    # Save the updated DataFrame to CSV
    df.to_csv('mohler_dataset.csv', index=False)
