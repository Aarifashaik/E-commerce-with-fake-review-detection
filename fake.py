import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

# Fake and real reviews dataset
data = {
    "review": [
        "This product is amazing! I'm very satisfied.", 
        "Worst product ever. Do not buy it!", 
        "Great service and fast delivery. Excellent product!", 
        "The product broke after one use. Very disappointed.",
        "Best purchase I have ever made!", 
        "Terrible quality, will never buy again."
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Genuine, 0 = Fake
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocess text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Remove stopwords
def preprocess_text(text):
    return ' '.join([word for word in text.lower().split() if word not in stop_words])

df['review'] = df['review'].apply(preprocess_text)

# Feature extraction using Bag-of-Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict if a review is fake or real
def predict_fake_review(review):
    review = preprocess_text(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return "Fake Review" if prediction == 0 else "Genuine Review"

# Test the prediction
test_review = "This product is fantastic!"
print(predict_fake_review(test_review))
