
# Spam Classifier

This project is a **Spam Classifier** built using machine learning to classify text messages as either "Spam" or "Ham" (non-spam). The project uses a pre-trained model and a GUI built with **Tkinter** for easy interaction. It also includes a logging feature to record predictions and a confusion matrix to evaluate the model's performance.

## Features

- **Message Classification**: Classifies messages as "Spam" or "Ham".
- **GUI Interface**: A user-friendly interface using **Tkinter** for easy interaction.
- **Prediction Logging**: Logs predictions with timestamps to a CSV file.
- **Prediction History**: Displays the last 5 predictions in the GUI.
- **Confusion Matrix**: Visualizes the model's performance using a heatmap.

## Requirements

Before running the project, make sure you have the following installed:

- Python 3.x
- Required libraries:
    - `pandas`
    - `joblib`
    - `tkinter`
    - `seaborn`
    - `matplotlib`
    - `scikit-learn`

You can install the dependencies using pip:

```bash
pip install pandas joblib seaborn matplotlib scikit-learn
```
## File Structure
```
Spam-Classifier/
│
├── model/
│   ├── spam_classifier_model.joblib    # Trained model file
│   └── tfidf_vectorizer.joblib        # TF-IDF vectorizer
│
├── data/
│   └── emailsmain.csv                 # Dataset for training and testing
│
├── spam_classifier_gui.py             # Main GUI script for the spam classifier
├── predictions_log.csv                # Log file to store predictions with timestamps
├── requirements.txt                  # List of required dependencies
└── README.md                         # Project documentation (Readme)
```

## Installation

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/letsconfuse/EmailSpamDetection.git
   cd EmailSpamDetection
   ```

2. **Download the model files**:
   Ensure that the model (`spam_classifier_model.joblib`) and the TF-IDF vectorizer (`tfidf_vectorizer.joblib`) files are placed in a folder named `model/` within your project directory.

3. **Run the application**:

   ```bash
   python gui.py
   ```

   This will start the Tkinter GUI, where you can input messages and classify them as "Spam" or "Ham".

## Usage

- **Message Input**: Type the message you want to classify in the text box.
- **Classify**: Click the "Classify" button to classify the message.
- **Clear**: Click the "Clear" button to clear the input field.
- **Prediction History**: The latest predictions will appear in the history section.
- **Confusion Matrix**: Click "Show Confusion Matrix" to view a heatmap of the model’s performance.

## Logging Format

Each prediction is logged in a CSV file (`predictions_log.csv`) with the following format:

```
Timestamp,Message,Label
2025-04-19 15:29:12,Dear Customer, we hope you're doing well!,1
2025-04-19 15:30:45,Your order has been shipped.,0
```

Where:
- `Timestamp`: Date and time when the message was classified.
- `Message`: The text of the message.
- `Label`: `1` for Spam, `0` for Ham.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Libraries**: This project uses Scikit-learn for the model and vectorizer, Tkinter for the GUI, Matplotlib and Seaborn for the confusion matrix visualization.
- **Dataset**: The spam detection model was trained on a labeled dataset of spam and ham emails.
