import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("data/iris.csv")
print(df.head())
print(df["species"].value_counts())

# Map species to numerical values
df["species"] = df["species"].map({
    "setosa":0,
    "versicolor":1,
    "virginica":2
})
print(df["species"].value_counts())

# Define Features and Label
X = df.drop("species", axis = 1)
y = df["species"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print(y_pred)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy is: {accuracy}")

# Save trained Model
joblib.dump(model,"model/model.pkl")