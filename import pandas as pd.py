import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('answers_similarity_scores.csv')

# Drop columns 'normalized_gpt_score' and 'gpt_similarity_score'
df = df.drop(['normalized_gpt_score', 'gpt_similarity_score'], axis=1)

# Save the updated DataFrame back to the CSV file
df.to_csv('answers_similarity_scores.csv', index=False)
