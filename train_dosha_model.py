import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    print("Loading synthetic dataset...")
    try:
        df = pd.read_csv('dosha_data.csv')
    except FileNotFoundError:
        print("Error: dosha_data.csv not found. Please run generate_dosha_dataset.py first.")
        return

    # Features and Target
    X = df.drop('target_dosha', axis=1)
    y = df['target_dosha']

    # Convert categorical text data into numbers (One-Hot Encoding)
    print("Encoding categorical variables...")
    X_encoded = pd.get_dummies(X)
    
    # Save the columns so the API knows the exact input structure expected
    model_columns = list(X_encoded.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("Saved model_columns.pkl")

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(clf, 'dosha_classifier.pkl')
    print("Successfully saved trained model to dosha_classifier.pkl")

if __name__ == "__main__":
    train_model()
