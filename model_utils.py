import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

MODEL_FILE = "student_model.pkl"
DATA_FILE = "student_data.csv"


def load_student_data():
    """Download the UCI Student Performance dataset and return its records."""
    student = fetch_ucirepo(id=320)
    data = pd.concat(
        [student.data.features, student.data.targets],
        axis=1
    )
    if "G3" not in data.columns:
        raise ValueError("The downloaded dataset does not contain the G3 column.")
    return data.drop_duplicates().reset_index(drop=True)


def build_model(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])


def train_student_model(data=None):
    """Train and evaluate the model. G3 is used only to create the target."""
    if data is None:
        data = load_student_data()

    data = data.drop_duplicates().reset_index(drop=True)
    y = (data["G3"] < 10).astype(int)
    X = data.drop(columns=["G3"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = build_model(X)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        output_dict=True,
        zero_division=0
    )

    joblib.dump(model, MODEL_FILE)
    return model, accuracy, report, X, data
