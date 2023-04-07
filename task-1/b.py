import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Loading data without header names bcz file doesn't have header names
data =pd.read_csv('data.csv')

# data = pd.DataFrame(data)
print(data)

print(data.describe())

print(data.isnull().sum())

# data Cleaning 
# Remove rows with missing values
data = data.dropna()

#  Add header names
column_names = ["native English speaker", "Course instructor", "Course", "semester", "Class size", "Class"]
data.columns = column_names

# Save data to new file
data.to_csv("modified_data.csv", index=False)
print(data)


corr = data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

#  Visualize distribution of target variable
sns.countplot(x="Class", data=data)
plt.show()
# 

# Feature engineering

# Create new feature for total score
data["total_score"] = data[["native English speaker", "Course instructor", "Course"]].sum(axis=1)

data["total_score_bins"] = pd.cut(data["total_score"], bins=[0, 50, 75, 100], labels=["Low", "Medium", "High"])

#Encode categorical variables using one-hot encoding

data = pd.get_dummies(data, columns=["semester", "total_score_bins"])

# Split data into features and target variable
X = data.drop(["Class"], axis=1)
y = data["Class"]


# Standardize features

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save preprocessed data to new file

data.to_csv("preprocessed_data.csv", index=False)



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading the preprocessed data

data = pd.read_csv("preprocessed_data.csv")

#  performing again Spliting data into features and target variable

X = data.drop(["Class"], axis=1)
y = data["Class"]

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train logistic regression model using grid search

param_grid = {
    "C": [0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

#Print best hyperparameters

print("Best hyperparameters:", grid_search.best_params_)

# Evaluate model on testing set

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# # Save model to local drive
import joblib
joblib.dump(grid_search, "teaching_assistant_model.joblib")
