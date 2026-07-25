# Ticket Classifier

A simple NLP model that reads support ticket text and predicts which department it belongs to — Billing, Technical, HR, or General. Built for the AI/ML Intern assessment.

## Files

- `ticket_classifier.py` — main script (loads data, trains model, evaluates, runs live demo)
- `tickets.csv` — labeled training dataset (64 sample tickets)

## Setup

```bash
pip install pandas scikit-learn
```

## How to run

```bash
python3 ticket_classifier.py
```

This will:
1. Load and clean the dataset
2. Train a Naive Bayes classifier on TF-IDF features
3. Print accuracy, precision/recall, and a confusion matrix
4. Predict categories for 5 new sample tickets
5. Test the 3 example tickets from the assessment page
6. Drop into a live demo where you can type your own ticket(s) and get an instant prediction

## Live demo

At the end of the script, type any ticket text and hit enter to classify it. You can also test several at once by separating them with commas:

```
Enter ticket(s): refund not received, app keeps crashing, need my leave balance
```

Type `exit` to quit.

## Approach

Cleaned the ticket text (lowercase, stripped punctuation/numbers), converted it to TF-IDF features using unigrams and bigrams with stopwords removed, then trained a Multinomial Naive Bayes classifier. NB was chosen because it's fast, works well on small text datasets, and is a standard baseline for this kind of classification task.

Every prediction returns a confidence score (via `predict_proba`). If confidence falls below 60%, the ticket is routed to `NEEDS HUMAN REVIEW` instead of forcing a guess — mirrors how a real triage system should behave when it's not sure. A simple keyword check also tags tickets as `URGENT` or `NORMAL` based on words like "down", "urgent", "crashing".

## What I'd improve with more time/data

The dataset is only 64 examples — Naive Bayes is frequency-based, so it would benefit a lot from more real ticket data, especially for categories like General that overlap in vocabulary with others. I'd also compare against Logistic Regression, which sometimes handles overlapping vocabulary better than NB's independence assumption. Given more time I'd add proper lemmatization instead of basic text cleaning, and evaluate on a larger held-out test set for a more reliable accuracy number than the current 13-ticket split.