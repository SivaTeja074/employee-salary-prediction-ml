import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# ✅ Load & clean data
# -------------------------------
data = pd.read_csv("C:/Users/tvtej/Downloads/employee-salary-prediction-ml/data/employee_salary_dataset.csv")


data['occupation'] = data['occupation'].replace({'?': 'Others'})
data['workclass'] = data['workclass'].replace({'?': 'NotListed'})
data['native-country'] = data['native-country'].replace({'?': 'Others'})

data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
data = data[data['occupation'] != 'Armed-Forces']
data = data[data['marital-status'] != 'Married-AF-spouse']
data = data[data['education'] != 'Preschool']
data = data[data['native-country'] != 'Holand-Netherlands']

data.drop(columns=['education'], inplace=True)

data = data[(data['age'] <= 75) & (data['age'] >= 17)]
data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
data = data[(data['hours-per-week'] <= 50) & (data['hours-per-week'] >= 35)]

# -------------------------------
# ✅ Define X, y
# -------------------------------
X = data.drop(columns=['income'])
y = data['income']

categorical_cols = [
    'workclass', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]
numeric_cols = [
    'age', 'fnlwgt', 'educational-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]

print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}
pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    pipelines[name] = pipe
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

best_model_name = max(results, key=results.get)
best_pipeline = pipelines[best_model_name]

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")

joblib.dump(best_pipeline, "best_model.pkl")
print("✅ Saved best pipeline as best_model.pkl")