import pandas as pd

# Paths to your dataset files
fake_path = r"F:\AI_SPAM\News _dataset\Fake.csv"
true_path = r"F:\AI_SPAM\News _dataset\True.csv"

# Load datasets
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Add label: 0 = real, 1 = fake
fake_df['label'] = 1
true_df['label'] = 0

# Combine title + text into single column
fake_df['text'] = fake_df['title'] + " " + fake_df['text']
true_df['text'] = true_df['title'] + " " + true_df['text']

# Keep only 'text' and 'label' columns
df = pd.concat([fake_df[['text','label']], true_df[['text','label']]], ignore_index=True)

# Save to train.csv in project folder
df.to_csv(r"F:\AI_SPAM\train.csv", index=False)
print("train.csv created successfully at F:\\AI_SPAM\\train.csv")
