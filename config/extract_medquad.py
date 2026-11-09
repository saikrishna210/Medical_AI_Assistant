import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

# Step 1: Dataset folder
dataset_path = Path("Dataset")

# Step 2: Output CSV file
output_path = Path("dataset/processed/medquad_processed.csv")

# Step 3: Create output directory
output_path.parent.mkdir(parents=True, exist_ok=True)

# Step 4: Initialize records
records = []

total_xml_files = 0
skipped_records = 0
failed_files = 0

# Step 5: Find all XML files
xml_files = list(dataset_path.rglob("*.xml"))

print("Total XML files found:", len(xml_files))

# Step 6: Process each XML file
for xml_file in xml_files:

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        focus = (root.findtext("Focus") or "").strip()
        source_url = (root.findtext("URL") or "").strip()

        # Step 7: Extract question-answer pairs
        for pair in root.iter("QAPair"):

            question_element = pair.find("Question")
            answer_element = pair.find("Answer")

            question = ""
            answer = ""

            if question_element is not None:
                question = "".join(question_element.itertext()).strip()

            if answer_element is not None:
                answer = "".join(answer_element.itertext()).strip()

            question_type = pair.get("qtype", "")

            # Step 8: Skip incomplete records
            if not question or not answer:
                skipped_records += 1
                continue

            # Step 9: Store record
            records.append({
                "id": len(records) + 1,
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "focus": focus,
                "source_url": source_url
            })
        total_xml_files += 1
    except ET.ParseError as error:
        failed_files += 1
        print("XML parsing error:", xml_file, error)

# Step 10: Write records into CSV
fieldnames = [
    "id",
    "question",
    "answer",
    "question_type",
    "focus",
    "source_url"
]

with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(records)

# Step 11: Display results
print("\n========== DATASET PROCESSING RESULTS ==========")

print("XML files found:", len(xml_files))
print("XML files processed:", total_xml_files)
print("XML files with parsing errors:", failed_files)
print("Valid question-answer records:", len(records))
print("Skipped incomplete records:", skipped_records)

print("\nCSV file created:", output_path)