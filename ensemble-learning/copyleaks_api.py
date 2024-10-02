import base64
import os
from dotenv import load_dotenv

import random
from copyleaks.copyleaks import Copyleaks
from copyleaks.exceptions.command_error import CommandError
from datetime import datetime

from copyleaks.models.submit.ai_detection_document import NaturalLanguageDocument, SourceCodeDocument
from copyleaks.models.export import *

# Register on https://api.copyleaks.com and grab your secret key (from the dashboard page).
load_dotenv()
EMAIL_ADDRESS = os.getenv("COPYLEAKS_EMAIL")
KEY = os.getenv("COPYLEAKS_API_KEY")

try:
    auth_token = Copyleaks.login(EMAIL_ADDRESS, KEY)
except CommandError as ce:
    response = ce.get_response()
    print(f"An error occurred (HTTP status code {response.status_code}):")
    print(response.content)
    exit(1)

print("Logged successfully!\nToken:")
print(auth_token)


def copyleaks_scan_text(text, filename):
    # print("Submitting a new file...")

    # This example is going to scan a FILE for plagiarism.
    # Alternatively, you can scan a URL using the class `UrlDocument`.
    scan_id = random.randint(100, 100000)  # generate a random scan id
    natural_language_submission = NaturalLanguageDocument(text)
    natural_language_submission.set_sandbox(True)
    response = Copyleaks.AiDetectionClient.submit_natural_language(auth_token, scan_id, natural_language_submission)

    ai_coverage = response.get("summary").get("ai") * 100

    # Extract and format the creation time
    creation_time = response.get('scannedDocument', {}).get("creationTime", "")
    formatted_date = datetime.strptime(creation_time.split('.')[0] + 'Z', '%Y-%m-%dT%H:%M:%SZ').strftime(
        '%m/%d/%Y %H:%M:%S')

    # Append the structured data to copyleaks_results
    results_json = {
        "Id": "d7czd4ni2re4adku",
        "Name": filename,
        "Date": formatted_date,
        "AI-Coverage": ai_coverage,
        "Plagiarism-Score": "-",
        "Report": ""
    }

    # copyleaks_results.append(results_json)

    # write this to file

    # Read the existing data from the JSON file
    file_path = 'copyleaks_results.json'

    copyleaks_content = []
    try:
        with open(file_path, 'r') as json_file:
            copyleaks_content = json.load(json_file)
    except FileNotFoundError:
        copyleaks_content = []  # Start with an empty list if the file does not exist

    # Append the new response
    copyleaks_content.append(results_json)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(copyleaks_content, json_file, indent=4)

    # print(f"New response added to {file_path}.")
    print(response.get("summary", {}).get("ai", 0) * 100)
    return response.get("summary", {}).get("ai", 0) * 100

copyleaks_scan_text("""

Neuroscience, the scientific study of the nervous system, has become a cornerstone in understanding complex human behaviors, brain functions, and various neurobiological disorders. The discipline integrates principles from biology, psychology, physics, and chemistry to explore the underlying mechanisms that drive cognition, perception, emotions, and behavior. Over the past few decades, advances in neuroscience have revolutionized medicine, education, and even artificial intelligence, demonstrating the profound implications of understanding the brain and nervous system.

One of the key contributions of neuroscience lies in its role in diagnosing and treating neurological and psychiatric disorders. Through neuroimaging techniques such as functional magnetic resonance imaging (fMRI) and positron emission tomography (PET), researchers have been able to identify abnormalities in brain structures and activity associated with conditions like Alzheimer's disease, Parkinson's disease, schizophrenia, and depression. These advancements allow for early diagnosis and intervention, significantly improving patient outcomes. Additionally, neuroscientific research has paved the way for the development of targeted therapies, such as deep brain stimulation for Parkinson’s disease and transcranial magnetic stimulation for depression, offering relief to patients where traditional treatments have failed.

In the realm of cognitive neuroscience, the study of how brain structures support mental functions has been particularly influential in education. Research on learning and memory has provided valuable insights into how the brain processes information, which in turn has informed the development of educational strategies. For example, findings on neuroplasticity— the brain's ability to reorganize itself by forming new neural connections—have led to teaching methods that encourage adaptive learning and cater to individual student needs. By understanding how the brain encodes, stores, and retrieves information, educators can create more effective learning environments and improve educational outcomes.

Moreover, neuroscience is also playing a critical role in shaping the field of artificial intelligence (AI). Neuroscientific insights into how the brain processes and learns from vast amounts of sensory input have influenced the design of neural networks and deep learning models, which are foundational to AI systems. Concepts such as pattern recognition, parallel processing, and hierarchical information processing in the brain are directly mirrored in machine learning algorithms. By continuing to draw on neuroscience, AI technologies can become more sophisticated, enabling more accurate simulations of human cognition and decision-making.

In conclusion, neuroscience has far-reaching implications that extend beyond its foundational understanding of the
brain. Its applications in diagnosing and treating neurological disorders, enhancing education, and advancing AI are
shaping various fields, making it an essential discipline for fostering innovation and improving human well-being. As
research continues to evolve, neuroscience will likely remain at the forefront of many scientific breakthroughs,
contributing to our understanding of both the human mind and the broader complexities of life.""", 'filename')
