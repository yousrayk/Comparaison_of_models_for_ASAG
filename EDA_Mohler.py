import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
ANSWERS_PATH = 'mohler_dataset.csv'
answers_data = pd.read_csv(ANSWERS_PATH)

# Calculate median and mean (average) of the 'score_avg' column
mean_score = answers_data['score_avg'].mean()
median_score = answers_data['score_avg'].median()

print(f"The mean (average) of the 'score_avg' column is: {mean_score}",f"The median of the 'score_avg' column is: {median_score}")

# Define intervals
intervals = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5),(4.5, 5.0)]

# Count occurrences within intervals
counts = [answers_data[(answers_data['score_avg'] >= start) & (answers_data['score_avg'] < end)].shape[0] for start, end in intervals]

# Plot a pie chart
fig, ax = plt.subplots()
ax.pie(counts, labels=[f'{start}-{end}' for start, end in intervals], autopct='%1.1f%%', startangle=90, counterclock=False)

# Set aspect ratio to be equal to ensure a circular pie chart
ax.axis('equal')

plt.title('Distribution of Assigned Grades', fontsize=14, weight='bold')
plt.show()

