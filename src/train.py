import sys
import os
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dvclive import Live
from sklearn.metrics import f1_score

# read the command line params
if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py features-dir-path model-filename\n'
    )
    sys.exit(1)

features_path = sys.argv[1]

params = yaml.safe_load(open('params.yaml'))['train']

n_estimators = params['n_estimators']

train_input_file = os.path.join(features_path, 'pr_train.csv')
test_input_file  = os.path.join(features_path, 'pr_test.csv')

# read the data from file
train_df = pd.read_csv(train_input_file)
test_df = pd.read_csv(test_input_file)

target = 'Transported'
features = list(train_df.columns)
features.remove(target)

X_train = train_df[features]
y_train = train_df[[target]]
X_test = test_df[features]
y_test = test_df[[target]]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
    
# with Live() as live:
#     live.log_param("n_estimators", n_estimators)
#     clf = RandomForestClassifier(n_estimators=n_estimators)
#     clf.fit(X_train, y_train.values.ravel())

#     y_train_pred = clf.predict(X_train)

#     live.log_metric("train/f1", f1_score(y_train, y_train_pred, average="weighted"), plot=False)
#     live.log_sklearn_plot(
#         "roc", y_train, y_train_pred, name="train/confusion_matrix",
#         title="Train Confusion Matrix")

#     y_test_pred = clf.predict(X_test)

#     live.log_metric("test/f1", f1_score(y_test, y_test_pred, average="weighted"), plot=False)
#     live.log_sklearn_plot(
#         "roc", y_test, y_test_pred, name="test/confusion_matrix",
#         title="Test Confusion Matrix")