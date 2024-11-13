import pandas as pd
import ast

df = pd.read_csv('dataset_segmented/valid/metadata.csv')

# Convert string representation of lists to actual lists
df['labels'] = df['labels'].apply(ast.literal_eval)

# Get the maximum length of any labels list to determine number of columns
max_labels = max(len(labels) for labels in df['labels'])

label_names = ['Crack', 'Red Dots', 'Toothmark']

for i in range(max_labels):
    df[label_names[i]] = df['labels'].apply(lambda x: x[i] if i < len(x) else None)

df = df.drop('labels', axis=1)
print(df.head())