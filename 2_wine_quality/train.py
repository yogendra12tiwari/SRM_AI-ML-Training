import pandas as pd
import joblib


df = pd.read_csv("Lab  winequality-red.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())


print("\nWine Quality Distribution:")
print(df["quality"].value_counts())

X = df.drop("quality", axis=1)
y = df["quality"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully!")