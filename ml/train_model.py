import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv("students.csv")

print("Dataset:")
print(data.head())

print("\nDataset shape:", data.shape)


# --------------------------------------------------
# 2. Features
# --------------------------------------------------

features = [
    "attendance",
    "sessional",
    "assignment",
    "class_performance",
    "viva",
    "mini_project",
    "lab_record",
    "study_hours"
]

X = data[features]

y = data["result"]


# --------------------------------------------------
# 3. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. ML Pipeline
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Test
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 7. Save model using Joblib
# --------------------------------------------------

joblib.dump(model, "student_model.joblib")

print("\nModel saved successfully!")
print("File: student_model.joblib")