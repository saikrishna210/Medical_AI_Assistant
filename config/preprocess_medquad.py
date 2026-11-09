import pandas as pd
import re
import html
from pathlib import Path

# Step 1: Define input and output paths
input_path = Path("dataset/processed/medquad_processed.csv")

output_path = Path("dataset/processed/medquad_cleaned.csv")

# Step 2: Load the original dataset
df = pd.read_csv(input_path)

print("Original dataset records:", len(df))

# Step 3: Function to clean text
def clean_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert value into string
    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)
    # converts html data into normal text

    # Replace multiple spaces and line breaks
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from beginning and end
    text = text.strip()

    return text


# Step 4: Clean question column
df["question_clean"] = df["question"].apply(clean_text)

# Step 5: Clean answer column
df["answer_clean"] = df["answer"].apply(clean_text)

# Step 6: Clean focus column
df["focus_clean"] = df["focus"].apply(clean_text)

# Step 7: Remove records with empty questions or answers
df = df[
    (df["question_clean"] != "") &
    (df["answer_clean"] != "")
].copy()

# Step 8: Check duplicate questions
duplicate_count = df["question_clean"].duplicated().sum()

print("Duplicate questions found:", duplicate_count)

# Step 9: Save cleaned dataset
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False, encoding="utf-8-sig")

# Step 10: Display results
print("\n========== PREPROCESSING RESULTS ==========")

print("Cleaned records:", len(df))

print("Total columns:", len(df.columns))

print("Output file:", output_path)

print("\nSample cleaned questions:")

for question in df["question_clean"].head(5):
    print("-", question)

print("\nSample cleaned answers:")

for answer in df["answer_clean"].head(3):
    print("-", answer[:200])
