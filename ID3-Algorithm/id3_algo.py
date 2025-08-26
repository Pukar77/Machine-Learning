import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ----- Custom Small Dataset -----
# Features: Age, Glucose, BloodPressure
# Target: Diabetes (Yes=1, No=0)
data = {
    "Age": [22, 25, 47, 52, 46, 56, 55, 60, 38, 42],
    "Glucose": [80, 85, 89, 120, 150, 200, 180, 210, 130, 140],
    "BloodPressure": [70, 72, 75, 90, 85, 95, 100, 110, 88, 92],
    "Diabetes": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

# ----- Decision Tree Classifier (ID3 using entropy) -----
clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
clf.fit(X, y)

# Predictions
y_pred = clf.predict(X)

# ----- Evaluation -----
print("Classification Report:\n", classification_report(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

# ----- Plot Decision Tree -----
plt.figure(figsize=(8,6))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("Decision Tree (ID3 - Entropy)")
plt.show()

# ----- Plot Confusion Matrix with Matplotlib -----
plt.matshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
