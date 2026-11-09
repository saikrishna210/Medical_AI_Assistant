import os
import xml.etree.ElementTree as ET

# Step 1: Define dataset location
dataset_path = "Dataset"

# Step 2: Find all XML files
xml_files = []

for root, directories, files in os.walk(dataset_path):

    for filename in files:

        if filename.lower().endswith(".xml"):

            xml_files.append(os.path.join(root, filename))

# Step 3: Display dataset information
print("Total XML files:", len(xml_files))

# Step 4: Check XML files
if len(xml_files) == 0:

    print("No XML files found. Check the dataset folder.")

else:

    # Step 5: Open first XML file
    file_path = xml_files[0]

    print("\nFirst XML file:", file_path)

    # Step 6: Parse XML
    tree = ET.parse(file_path)

    root_element = tree.getroot()

    # Step 7: Display root element
    print("\nRoot Element:", root_element.tag)

    # Step 8: Display medical topic
    print("Medical Focus:", root_element.findtext("Focus"))

    # Step 9: Find question-answer records
    qa_pairs = list(root_element.iter("QAPair"))

    print("\nTotal Q&A records:", len(qa_pairs))

    # Step 10: Display first question and answer
    if qa_pairs:

        first_pair = qa_pairs[0]

        question = first_pair.findtext("Question")
        answer = first_pair.findtext("Answer")

        print("\nFirst Question:")
        print(question)

        print("\nFirst Answer:")
        print(answer)