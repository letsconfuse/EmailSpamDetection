import tkinter as tk
from tkinter import messagebox
import pandas as pd
from joblib import load
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load your models and vectorizer
model = load('model/spam_classifier_model.joblib')
vectorizer = load('model/tfidf_vectorizer.joblib')

# Global variable for storing prediction history
prediction_history = []

# Function to classify the input message
def classify_message():
    # Get the user input
    user_message = message_entry.get("1.0", "end-1c").strip()

    if user_message == "":
        result_label.config(text="Please enter a message to classify.", fg="red")
        return
    
    # Transform the input message using the vectorizer and predict
    transformed_message = vectorizer.transform([user_message])
    prediction = model.predict(transformed_message)

    # Display result on the GUI instead of an alert window
    if prediction[0] == 0:
        result_label.config(text="This is Ham!", fg="green")
    else:
        result_label.config(text="This is Spam!", fg="red")

    # Log the prediction
    log_prediction(user_message, prediction[0])

# Function to clear the input and result label
def clear_input():
    message_entry.delete("1.0", "end-1c")
    result_label.config(text="")
    clear_feedback.config(text="Input cleared.", fg="blue")

# # Function to log predictions into a CSV
# def log_prediction(message, prediction):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     prediction_label = "Spam" if prediction == 1 else "Ham"

#     # Clean message text by stripping unwanted characters (like extra quotes)
#     message = message.replace('""', '').replace('"', '').strip()
#     message = message.replace('\n', ' ').replace('\r', ' ').strip()

#     # Log to CSV
#     log_file_path = 'predictions_log.csv'
#     file_exists = os.path.isfile(log_file_path)
#     log_entry = pd.DataFrame([{'Timestamp': timestamp, 'Message': message, 'Prediction': prediction_label}])
#     log_entry.to_csv(log_file_path, mode='a', header=not file_exists, index=False)

#     # Save to history for GUI display
#     prediction_history.insert(0, (timestamp, message, prediction_label))

#     # Update the history label with the latest prediction
#     update_history()

# Function to log predictions into a CSV with the specified format
def log_prediction(message, prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    category = "spam" if prediction == 1 else "ham"
    label = 1 if prediction == 1 else 0

    # Clean message text by stripping unwanted characters (like extra quotes)
    message = message.replace('""', '').replace('"', '').strip()
    message = message.replace('\n', ' ').replace('\r', ' ').strip()

    # Log to CSV
    log_file_path = 'predictions_log.csv'
    file_exists = os.path.isfile(log_file_path)

    # Prepare log entry in the desired format
    log_entry = pd.DataFrame([{
        'Timestamp': timestamp,
        'Category': category,
        'Message': message,
        'Label': label
    }])

    # Append to CSV
    log_entry.to_csv(log_file_path, mode='a', header=not file_exists, index=False)

    # Save to history for GUI display
    prediction_history.insert(0, (timestamp, category, message, label))

    # Update the history label with the latest prediction
    update_history()


# Function to update the prediction history display
def update_history():
    history_text.delete(1.0, tk.END)  # Clear previous history
    for entry in prediction_history[:5]:  # Limit to last 5 predictions
        history_text.insert(tk.END, f"{entry[0]} - {entry[1]} -> {entry[2]}\n")

# Function to display confusion matrix as a heatmap
def display_confusion_matrix():
    # Load the dataset and preprocess (similar to your script)
    data = pd.read_csv('data/emailsmain.csv')
    X = data['Message']
    y = data['Label']

    # Transform text using the vectorizer
    X_tfidf = vectorizer.transform(X)

    # Predict using the model
    y_pred = model.predict(X_tfidf)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Set up the main window
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("500x500")
root.config(bg="#F0F0F0")

# Add a title label
title_label = tk.Label(root, text="Email Spam Classifier", font=("Arial", 16), bg="#F0F0F0")
title_label.pack(pady=10)

# Add a text box for message input
message_entry = tk.Text(root, height=5, width=40)
message_entry.pack(pady=10)

# Add a button to classify the message
classify_button = tk.Button(root, text="Classify", width=20, command=classify_message, bg="#4CAF50", fg="white")
classify_button.pack(pady=5)

# Add a button to clear the input
clear_button = tk.Button(root, text="Clear", width=20, command=clear_input, bg="#FF5733", fg="white")
clear_button.pack(pady=5)

# Label to display the classification result
result_label = tk.Label(root, text="", font=("Arial", 12), bg="#F0F0F0")
result_label.pack(pady=10)

# Label for clear feedback
clear_feedback = tk.Label(root, text="", font=("Arial", 10), bg="#F0F0F0")
clear_feedback.pack(pady=5)

# Add a button to display confusion matrix
conf_matrix_button = tk.Button(root, text="Show Confusion Matrix", width=20, command=display_confusion_matrix, bg="#3498DB", fg="white")
conf_matrix_button.pack(pady=5)

# Add a frame for displaying prediction history
history_frame = tk.Frame(root)
history_frame.pack(pady=10, fill="both", expand=True)

history_label = tk.Label(history_frame, text="Prediction History", font=("Arial", 14), bg="#F0F0F0")
history_label.pack()

history_text = tk.Text(history_frame, height=6, width=50)
history_text.pack()

# Run the GUI loop
root.mainloop()
