import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Load your dataset
data = pd.read_csv('data/emailsmain.csv')  # Ensure this file is in the same directory

# Preprocessing (assuming you have already cleaned your data)
X = data['Message']
y = data['Label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model/spam_classifier_model.joblib')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')

# Test and print classification metrics
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {model.score(X_test_tfidf, y_test)}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- New Features: Visualization and Logging ---

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# # Function to log predictions and inputs to a CSV
# def log_predictions(message, prediction):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # Clean message text by stripping unwanted characters (like extra quotes)
#     message = message.replace('""', '').replace('"', '').strip()

#     # Replace newlines or carriage returns with a space or custom delimiter
#     message = message.replace('\n', ' ').replace('\r', ' ').strip()
    

#     # Prepare log entry
#     log_entry = {'Timestamp': timestamp, 
#                  'Message': message, 
#                  'Prediction': prediction}

#     # Check if file exists
#     log_file_path = 'predictions_log.csv'
#     file_exists = os.path.isfile(log_file_path)

#     # Append to CSV
#     log_df = pd.DataFrame([log_entry])
#     log_df.to_csv(log_file_path, mode='a', index=False, header=not file_exists)


# Function to log predictions and inputs to a CSV
def log_predictions(message, prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clean message text by stripping unwanted characters (like extra quotes)
    message = message.replace('""', '').replace('"', '').strip()

    # Replace newlines or carriage returns with a space
    message = message.replace('\n', ' ').replace('\r', ' ').strip()

    # Map prediction (1/0) to category
    category = 'spam' if prediction == 1 else 'ham'

    # Prepare log entry with desired columns
    log_entry = {
        'Timestamp': timestamp,
        'Category': category,
        'Message': message,
        'Label': prediction
    }

    # Check if file exists
    log_file_path = 'predictions_log.csv'
    file_exists = os.path.isfile(log_file_path)

    # Append to CSV
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(log_file_path, mode='a', index=False, header=not file_exists)


# Example usage of logging function with the long message:
long_message = '''Dear Customer,

We hope you're doing well! As one of our valued customers, we are pleased to offer you exclusive discounts on a variety of products and services. Whether you're looking for the latest gadgets, household essentials, or fashion items, we've got something special just for you.

Our summer sale is now live, and you can save up to 50% on select items. This is a limited-time offer, so make sure you grab your favorites before they're gone! To access the sale, simply click the link below and start shopping.

Click here to claim your discount now!

In addition, we're offering free shipping on all orders over $50, and if you refer a friend, both you and your friend will receive an additional 10% off on your next purchase. It's our way of saying thank you for being such a loyal customer.

To make things even easier, we've created a personalized shopping experience just for you. By visiting our website, you can view recommended products based on your past purchases, browse the latest trends, and even track your order in real-time.

We look forward to serving you and hope you enjoy these amazing offers!

Best regards,
The Sales Team
XYZ Corporation
Contact us at: support@xyzcorporation.com
Follow us on social media for the latest updates: Facebook, Instagram, Twitter.'''

# Predict and log the long message
sample_prediction = model.predict(vectorizer.transform([long_message]))[0]
log_predictions(long_message, sample_prediction)
