from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from copyleaks_results import copyleaks_results
from copyleaks_api import copyleaks_scan_text

import os

import numpy as np
import sklearn
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

import main as aux_function

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

HUMAN = 0
AI = 1


def generate_training_xy(dir_name: str, expected_value: int) -> tuple[list[list], list[int]]:
    """ Generate training x and y values using the files in the given directory"""
    x_results = []
    y_results = []

    # Iterate through files in the directory
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)

        # Check if it is a file (and not a directory)
        if os.path.isfile(file_path) and not filename.startswith('.'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()

                # Process the file contents
                text_feature = aux_function.get_text_features(text)
                text_feature.append(copyleaks_scan_text(text, filename))

                x_results.append(text_feature)
                y_results.append(expected_value)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return x_results, y_results


def get_copyleaks_results(text, filename: str):
    """Temp function only """
    # temp function only
    for record in copyleaks_results:
        if record["Name"].lower() == filename.lower():
            return record["AI-Coverage"] / 100

    print("cant find copyleaks for file " + filename)


def perform_testing(dir_name: str, model: VotingClassifier) -> list[int]:
    """ Performs testing on the files in the given directory"""
    X_test = []
    filenames = []

    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)

        # Check if it is a file
        if os.path.isfile(file_path) and not filename.startswith('.'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                text_feature = aux_function.get_text_features(text)
                text_feature.append(copyleaks_scan_text(text, filename))
                X_test.append(text_feature)
                filenames.append(filename)

    results = model.predict_proba(X_test)

    return results, filenames


def ensemble():
    """
        The main ensemble function
    """
    # Create the individual classifiers
    lr = LogisticRegression(random_state=42, max_iter=1000)
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier(random_state=42)

    # Create the ensemble classifier
    ensemble_model = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('tree', tree)], voting='soft')

    X_train = []
    Y_train = []

    # generate training data
    for dir_name, label in [("training-ai", AI), ("training-human", HUMAN)]:
        train_x, train_y = generate_training_xy(dir_name, label)
        X_train += train_x
        Y_train += train_y

    # perform training
    ensemble_model.fit(X_train, Y_train)

    # perform testing
    ai_test_results, ai_filenames = perform_testing("test-ai", ensemble_model)
    human_test_results, human_filenames = perform_testing("test-human", ensemble_model)

    # results output
    print("filename", "results", "expected")

    for i in range(len(ai_test_results)):
        print(f"{ai_filenames[i]}\t{ai_test_results[i]}\tAI")

    # Loop through Human test results
    for i in range(len(human_test_results)):
        print(f"{human_filenames[i]}\t{human_test_results[i]}\tHUMAN")

    # output stats
    calculate_overall_stats(ai_test_results.tolist(), human_test_results.tolist())
    calculate_human_only_stats(human_test_results.tolist())
    calculate_ai_only_stats(ai_test_results.tolist())

    return


def calculate_overall_stats(ai_test_results, human_test_results):
    print("\n --- Overall Stats --- ")
    y_results = ai_test_results + human_test_results
    y_true = [AI] * len(ai_test_results) + [HUMAN] * len(human_test_results)
    y_pred = [sublist[1] for sublist in y_results]
    calc_stats_binary(y_true, y_pred, 0.5)
    calc_stats(y_true, y_pred)


def calculate_human_only_stats(human_test_results):
    print("\n --- Human-Only Stats --- ")
    y_true = [1] * len(human_test_results)
    y_pred = [sublist[0] for sublist in human_test_results]

    calc_stats_binary(y_true, y_pred, 0.5)
    calc_stats(y_true, y_pred)


def calculate_ai_only_stats(ai_test_results):
    print("\n --- AI-Only Stats --- ")
    y_true = [AI] * len(ai_test_results)
    y_pred = [sublist[1] for sublist in ai_test_results]
    calc_stats_binary(y_true, y_pred, 0.5)
    calc_stats(y_true, y_pred)


def read_file_to_text(file_path):
    """
        Read file from the file_path and convert it to plain 'utf-8' text
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    # Read the file contents
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    return text


# print(main.get_text_features(read_file_to_text("training-ai/13.txt")))


def calc_stats(y_true, y_pred):
    """
     Calculate mae and mse based on the y_true and y_pred values
    :param y_true:
    :param y_pred:
    :return:
    """
    import sklearn.metrics as metrics

    # Assuming y_test and y_pred are already defined
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    # Print the metrics in a nicely formatted manner
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE) : {mse:.2f}")
    # print(f"R-squared (R²)           : {r2:.2f}")

    return


def calc_stats_binary(y_true, y_pred_prob, threshold):
    """Calculate the function performance in binary terms.
        All y_pred_prob values will be rounded to 0 or 1 based on the
        "threshold" value
    """
    y_prob = [1 if prob >= threshold else 0 for prob in y_pred_prob]

    accuracy = aux_function.accuracy_score(y_true, y_prob)
    precision = aux_function.precision_score(y_true, y_prob)
    recall = aux_function.recall_score(y_true, y_prob)
    f1 = aux_function.f1_score(y_true, y_prob)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def main():
    ensemble()


if __name__ == "__main__":
    main()
