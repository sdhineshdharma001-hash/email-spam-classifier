import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# 2. Select required columns
if "v1" in data.columns and "v2" in data.columns:
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]
else:
    data = data.iloc[:, :2]
    data.columns = ["label", "message"]

# 3. Remove empty rows
data = data.dropna()

# 4. Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove any rows with unknown labels
data = data.dropna()

# 5. Split data
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 7. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 8. Test model
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("================================")
print("EMAIL SPAM CLASSIFIER")
print("================================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print()
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# 9. Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 10. Save vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print("Training completed successfully!")