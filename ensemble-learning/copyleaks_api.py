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

