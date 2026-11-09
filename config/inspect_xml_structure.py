import xml.etree.ElementTree as ET
from pathlib import Path

dataset_path = Path("Dataset")

xml_files = list(dataset_path.rglob("*.xml"))

print("Total XML files:", len(xml_files))

found = False

for file_path in xml_files:

    tree = ET.parse(file_path)
    root = tree.getroot()

    for pair in root.iter("QAPair"):

        question_element = pair.find("Question")
        answer_element = pair.find("Answer")

        question = ""
        answer = ""

        if question_element is not None:
            question = "".join(question_element.itertext()).strip()

        if answer_element is not None:
            answer = "".join(answer_element.itertext()).strip()

        if question and answer:

            print("\nXML File:", file_path)
            print("\nRoot Tag:", root.tag)

            print("\nFocus:", root.findtext("Focus"))

            print("\nQAPair Attributes:", pair.attrib)

            print("\nQuestion:", question)

            print("\nAnswer:", answer[:500])

            found = True
            break

    if found:
        break

if not found:
    print("No XML file with a non-empty answer was found.")