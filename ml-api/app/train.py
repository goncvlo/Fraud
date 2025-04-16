import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# load and transform data
df = pd.read_csv('../creditcard_2023.csv')
df = df[['V4', 'V14', 'Class']].rename(columns={'Class': 'label'})

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['label']), df['label']
    , test_size=0.1
    , random_state=42
    , shuffle=True
    , stratify=df['label']
    )

# train model and evaluate it
clf = XGBClassifier(max_depth=20, random_state=42)
clf.fit(X_train, y_train)

test_score = clf.score(X_test, y_test)

# save model and metric
joblib.dump(clf, 'artifacts/model.joblib')
joblib.dump(test_score, 'artifacts/test_score.joblib')
