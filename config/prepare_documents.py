
import pandas as pd
from pathlib import Path

# Step 1: Find the main project folder
project_root = Path(__file__).resolve().parents[2]

print("Project folder:", project_root)

# Step 2: Search for the cleaned CSV file
matches = list(
    project_root.rglob("medquad_cleaned.csv")
)

if not matches:

    print("Error: medquad_cleaned.csv was not found!")

    print("\nCSV files found in the project:")

    for file in project_root.rglob("*.csv"):
        print(file)

else:

    # Step 3: Select the cleaned dataset
    input_path = matches[0]

    print("Dataset found:", input_path)

    # Step 4: Define output path in the same folder
    output_path = input_path.parent / "medquad_documents.csv"

    # Step 5: Load dataset
    df = pd.read_csv(input_path)

    print("\nTotal records loaded:", len(df))

    # Step 6: Create combined medical documents
    df["document"] = (
        "Medical Topic: " + df["focus_clean"].fillna("").astype(str) +
        "\nQuestion: " + df["question_clean"].fillna("").astype(str) +
        "\nAnswer: " + df["answer_clean"].fillna("").astype(str)
    )

    # Step 7: Remove empty questions and answers
    df = df[
        (df["question_clean"].fillna("").str.strip() != "") &
        (df["answer_clean"].fillna("").str.strip() != "")
    ].copy()

    # Step 8: Select required columns
    documents = df[
        [
            "id",
            "focus_clean",
            "question_clean",
            "answer_clean",
            "source_url",
            "document"
        ]
    ]

    # Step 9: Save prepared documents
    documents.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Step 10: Display results
    print("\n========== DOCUMENT PREPARATION RESULTS ==========")

    print("Total documents:", len(documents))

    print("\nFirst prepared document:")
    print(documents.iloc[0]["document"])

    print("\nCSV file created successfully!")
    print("Output location:", output_path)
