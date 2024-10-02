import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
import textstat

nltk.download('punkt')
nltk.download('stopwords')

# Load GPT-2 model for perplexity calculation
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def read_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts


# def calculate_readability_score(text):
#     sentences = sent_tokenize(text)
#     words = word_tokenize(text)
#     if not sentences or not words:
#         return 0
#     avg_sentence_length = len(words) / len(sentences)
#     avg_syllables_per_word = sum(count_syllables(word) for word in words) / len(words)
#     return 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

def calculate_readability_score(text):
    return textstat.flesch_reading_ease(text)


def count_syllables(word):
    return max(1, len([vowel for vowel in word if vowel in 'aeiou']))


def calculate_perplexity(text, model=gpt2_model, tokenizer=gpt2_tokenizer, max_length=1024):
    encodings = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = encodings.input_ids[:, :max_length]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss).item()


def calculate_lexical_density(text):
    words = word_tokenize(text.lower())
    content_words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    return len(content_words) / len(words) if words else 0


def calculate_avg_word_length(text):
    words = word_tokenize(text.lower())
    return sum(len(word) for word in words) / len(words) if words else 0


def calculate_ngram_diversity(text, n=3):
    tokens = word_tokenize(text.lower())
    n_grams = list(ngrams(tokens, n))
    return len(set(n_grams)) / len(n_grams) if n_grams else 0


def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences) if sentences else 0


def get_text_features(text):
    return [
        calculate_readability_score(text),
        calculate_perplexity(text),
        calculate_lexical_density(text),
        calculate_avg_word_length(text),
        calculate_ngram_diversity(text),
        calculate_avg_sentence_length(text)
    ]


def process_directory(directory, label):
    texts = read_files(directory)
    features = [get_text_features(text) for text in texts]
    return pd.DataFrame(features,
                        columns=['readability', 'perplexity', 'lexical_density', 'avg_word_length', 'ngram_diversity',
                                 'avg_sentence_length']), pd.Series([label] * len(texts))


def train_and_save_model(ai_directory, human_directory, model_filename):
    print("Processing AI-generated texts...")
    ai_features, ai_labels = process_directory(ai_directory, 1)

    print("Processing human-written texts...")
    human_features, human_labels = process_directory(human_directory, 0)

    X = pd.concat([ai_features, human_features])
    y = pd.concat([ai_labels, human_labels])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    with open(model_filename, 'wb') as file:
        pickle.dump((model, X.columns.tolist()), file)
    print(f"\nModel and feature names saved to {model_filename}")

    return model, X.columns.tolist()


def load_model(filename):
    with open(filename, 'rb') as file:
        model, feature_names = pickle.load(file)
    print(f"Model and feature names loaded from {filename}")
    return model, feature_names


def analyze_feature_importance(model, features, feature_names):
    coefficients = model.coef_[0]
    contributions = coefficients * features
    feature_contributions = list(zip(feature_names, contributions))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_contributions


def interpret_contributions(feature_contributions, prediction):
    total_contribution = sum(abs(contrib) for _, contrib in feature_contributions)
    interpreted_contributions = []
    for feature, contribution in feature_contributions:
        percentage = (contribution / total_contribution) * 100
        if (prediction == "AI-generated" and contribution > 0) or (prediction == "Human-written" and contribution < 0):
            direction = "towards this prediction"
        else:
            direction = "against this prediction"
        interpreted_contributions.append((feature, abs(percentage), direction))
    return interpreted_contributions


def classify_text(text, model, feature_names):
    features = get_text_features(text)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]
    feature_contributions = analyze_feature_importance(model, features, feature_names)
    prediction_label = "AI-generated" if prediction == 1 else "Human-written"
    interpreted_contributions = interpret_contributions(feature_contributions, prediction_label)
    return prediction_label, probability, interpreted_contributions


if __name__ == "__main__":
    ai_directory = "./data/ai"
    human_directory = "./data/human_samples"
    model_filename = "ai_detection_model.pkl"
    train_and_save_model(ai_directory, human_directory, model_filename)

    loaded_model, feature_names = load_model(model_filename)

    test_texts = [
        "This is a human-written test sentence. It's not very long, but it should be enough for a quick test.",
        "The solar system is a vast expanse centered around our Sun, a medium-sized star in the Milky Way galaxy. It "
        "consists of eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune, orbiting the "
        "Sun at varying distances. These planets are accompanied by numerous moons, asteroids, comets, "
        "and dwarf planets like Pluto. The inner planets - Mercury, Venus, Earth, and Mars - are rocky, "
        "while the outer planets - Jupiter, Saturn, Uranus, and Neptune - are gas giants. The solar system is held "
        "together by the Sun's gravitational pull, with each celestial body following its own orbital path. This "
        "cosmic neighborhood extends far beyond the planets, encompassing the Kuiper Belt and the theoretical Oort "
        "Cloud at its outer reaches.",
    ]

    print("\nClassifying test texts:")
    for i, text in enumerate(test_texts):
        prediction, probability, feature_contributions = classify_text(text, loaded_model, feature_names)
        print(f"\nText {i + 1}:")
        print(f"Content: '{text}'")
        print(f"Prediction: {prediction}")
        print(f"Probability of being AI-generated: {probability:.4f}")
        print("Main factors contributing to this decision:")
        for feature, contribution, direction in feature_contributions[:3]:  # Show top 3 contributing factors
            print(f"  {feature}: {contribution:.2f}% ({direction})")
