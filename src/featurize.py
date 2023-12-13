import sys
import os
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# read command line params
if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython featurize.py data-dir-path features-dir-path\n'
    )
    sys.exit(1)

data_path = sys.argv[1]
features_path = sys.argv[2]

os.makedirs(features_path, exist_ok=True)

train_input_file = os.path.join(data_path, 'train.csv')
test_input_file  = os.path.join(data_path, 'test.csv')

# read the data from file
train_df = pd.read_csv(train_input_file)
test_df = pd.read_csv(test_input_file)

cat_df = train_df.select_dtypes(include=['object'])
num_df = train_df.select_dtypes(exclude=['object'])

train_df['Age'] = train_df['Age'].transform(lambda x: x.fillna(x.mean()))
test_df['Age'] = test_df['Age'].transform(lambda x: x.fillna(x.mean()))

num_сols_with_missing = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_df[num_сols_with_missing] = train_df[num_сols_with_missing].transform(lambda x: x.fillna(x.mode()))
test_df[num_сols_with_missing] = test_df[num_сols_with_missing].transform(lambda x: x.fillna(x.mode()))

cat_сols_with_missing = ['HomePlanet', 'Destination']
train_df[cat_сols_with_missing] = train_df[cat_сols_with_missing].transform(lambda x: x.fillna(x.dropna().mode()[0]))
test_df[cat_сols_with_missing] = test_df[cat_сols_with_missing].transform(lambda x: x.fillna(x.dropna().mode()[0]))

train_df = train_df.dropna()
test_df = test_df.dropna()

for col, upper_bound in (
    ('RoomService', 5000),
    ('FoodCourt', 20000),
    ('ShoppingMall', 5000),
    ('Spa', 10000),
    ('VRDeck', 14000),
):
    train_df = train_df.drop(train_df[train_df[col] > upper_bound].index)

cat_df = train_df.select_dtypes(include=['object'])
cat_df_ = test_df.select_dtypes(include=['object'])

ohe_df = pd.get_dummies(cat_df[['HomePlanet', 'Destination']])
ohe_df_ = pd.get_dummies(cat_df_[['HomePlanet', 'Destination']])

label_encoder = LabelEncoder()

encoded_neigh = pd.Series(label_encoder.fit_transform(train_df['Cabin']))
encoded_neigh = encoded_neigh.rename("Cabin")

encoded_neigh_ = pd.Series(label_encoder.fit_transform(test_df['Cabin']))
encoded_neigh_ = encoded_neigh.rename("Cabin")

num_df = train_df.select_dtypes(exclude=['object'])
train_df = pd.concat([ohe_df, num_df, encoded_neigh], axis=1, join='inner')
test_df = pd.concat([ohe_df_, num_df, encoded_neigh_], axis=1, join='inner')

data_path_ = os.path.join('data', 'features')

train_df.to_csv(os.path.join(data_path_, "pr_train.csv"))
test_df.to_csv(os.path.join(data_path_, "pr_test.csv"))