import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loading the dataset
df = pd.read_csv("tickets.csv")
print("Loaded", len(df), "tickets")
print(df["label"].value_counts())

# clean up the text a bit
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove numbers/punctuation
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# tfidf with bigrams, removing stopwords
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train the model
model = MultinomialNB(alpha=0.3)
model.fit(X_train_vec, y_train)

# check how it did
predictions = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("Labels order:", model.classes_)

# function to classify a new ticket, with confidence + review threshold
REVIEW_THRESHOLD = 0.60
URGENT_WORDS = ["down", "urgent", "not working", "crashing", "asap", "immediately"]

def classify_ticket(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)

    if confidence < REVIEW_THRESHOLD:
        pred = "NEEDS HUMAN REVIEW"

    priority = "URGENT" if any(word in text.lower() for word in URGENT_WORDS) else "NORMAL"

    return pred, round(confidence * 100, 1), priority

# some new tickets i wrote to test it
new_tickets = [
    "my payment is stuck and i need a refund urgently",
    "the app crashed twice, its not working at all",
    "can you tell me my leave balance",
    "just wanted to know your business hours",
    "server has been down since this morning, please fix asap",
]

print("\nPredictions on new tickets:")
for ticket in new_tickets:
    category, confidence, priority = classify_ticket(ticket)
    print(f"'{ticket}'")
    print(f"  -> {category} | confidence: {confidence}% | priority: {priority}\n")

# checking against the sample tickets given in the assessment
sample_tickets = [
    "refund delayed",
    "API returns 500 error",
    "invoice not received",
]

print("Sample tickets from assessment:")
for ticket in sample_tickets:
    category, confidence, priority = classify_ticket(ticket)
    print(f"'{ticket}' -> {category} | confidence: {confidence}% | priority: {priority}")

# live demo, you can paste multiple tickets separated by commas to test several at once
print("\n--- Live demo ---")
print("Type a ticket and press enter. For multiple, separate with commas.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter ticket(s): ")
    if user_input.strip().lower() == "exit":
        break

    entries = [t.strip() for t in user_input.split(",") if t.strip()]
    for entry in entries:
        category, confidence, priority = classify_ticket(entry)
        print(f"  '{entry}' -> {category} | confidence: {confidence}% | priority: {priority}")