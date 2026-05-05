import spacy
from sklearn.metrics import classification_report, accuracy_score


nlp = spacy.load("en_core_web_sm")


texts = [
    "Apple is opening a new office in Mumbai",
    "Elon Musk is the CEO of Tesla"
]


true_entities = [
    [("Apple", "ORG"), ("Mumbai", "GPE")],
    [("Elon Musk", "PERSON"), ("Tesla", "ORG")]
]



y_true = []
y_pred = []

for text, true_ents in zip(texts, true_entities):
    doc = nlp(text)
    pred_ents = [(ent.text, ent.label_) for ent in doc.ents]

    for ent_text, ent_label in true_ents:
        y_true.append(ent_label)

        if (ent_text, ent_label) in pred_ents:
            y_pred.append(ent_label)
        else:
            y_pred.append("O")  


print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nReport:\n", classification_report(y_true, y_pred))