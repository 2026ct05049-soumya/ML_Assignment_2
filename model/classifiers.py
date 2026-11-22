"""Build independent preprocessing and classification pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]


def build_models():
    classifiers = [
        ("logistic_regression", "Logistic Regression", LogisticRegression(max_iter=1000)),
        ("decision_tree", "Decision Tree", DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=SEED)),
        ("knn", "kNN", KNeighborsClassifier(n_neighbors=15)),
        ("gaussian_nb", "Naive Bayes", GaussianNB()),
        ("random_forest", "Random Forest (Ensemble)", RandomForestClassifier(
            n_estimators=120, max_depth=18, min_samples_leaf=4, random_state=SEED, n_jobs=4,
        )),
    ]
    for key, name, classifier in classifiers:
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if key in {"logistic_regression", "knn", "gaussian_nb"}:
            numeric_steps.append(("scale", StandardScaler()))
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocess = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ])
        yield key, name, Pipeline([("preprocess", preprocess), ("classifier", classifier)])
