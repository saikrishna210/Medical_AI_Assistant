
import pandas as pd

# Step 1: Load cleaned dataset
file_path = "dataset/processed/medquad_cleaned.csv"

df = pd.read_csv(file_path)

print("Total records:", len(df))

# Step 2: Group records by cleaned question
question_groups = (
    df.groupby("question_clean")
    .agg(
        total_records=("id", "count"),
        unique_answers=("answer_clean", "nunique")
    )
    .reset_index()
)

# Step 3: Find repeated questions
duplicates = question_groups[
    question_groups["total_records"] > 1
]

print("\nTotal repeated questions:", len(duplicates))

# Step 4: Find questions with different answers
different_answers = duplicates[
    duplicates["unique_answers"] > 1
]

print(
    "\nRepeated questions with different answers:",
    len(different_answers)
)

# Step 5: Display sample duplicate questions
print("\nSample repeated questions:")

print(duplicates.head(10).to_string(index=False))

# Step 6: Save duplicate report
duplicates.to_csv(
    "dataset/processed/duplicate_report.csv",
    index=False
)

# Step 7: Save questions having different answers
different_answers.to_csv(
    "dataset/processed/different_answers_report.csv",
    index=False
)

print("\nDuplicate analysis completed!")
