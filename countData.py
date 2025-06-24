import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\ASUS\\Desktop\\Mini Project 2\\Suicide_Detection.csv')  # Update with the correct file path

# Check the counts of 'non-suicidal' and 'suicidal' in the 'class' column
class_counts = df['class'].value_counts()

print("Class Distribution:")
print(class_counts)
