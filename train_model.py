from model_utils import load_student_data, train_student_model, DATA_FILE

print("Downloading the UCI Student Performance dataset...")
data = load_student_data()
data.to_csv(DATA_FILE, index=False)

print("\nDataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nColumn data types:")
print(data.dtypes)
print("\nMissing values per column:")
print(data.isnull().sum())

print("\nTraining the Random Forest model...")
model, accuracy, report, X, data = train_student_model(data)

print("\nModel evaluation")
print("Accuracy:", round(accuracy, 3))
print("\nClassification report:")
for label in ["Not at risk", "At risk"]:
    values = report[label]
    print(
        f"{label}: precision={values['precision']:.3f}, "
        f"recall={values['recall']:.3f}, "
        f"F1-score={values['f1-score']:.3f}"
    )

print("\nFiles created:")
print("- student_data.csv")
print("- student_model.pkl")
