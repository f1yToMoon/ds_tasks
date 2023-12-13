import pandas as pd
import os
import yaml

# params = yaml.safe_load(open('params.yaml'))['prepare']

# categories = params['categories']

# create folder to save file
data_path = os.path.join('data', 'prepared')
os.makedirs(data_path, exist_ok=True)

train_df = pd.read_csv("C:/Users/Никита/DS-task/train.csv")
test_df = pd.read_csv("C:/Users/Никита/DS-task/test.csv")

train_df.drop(['PassengerId', 'Name', 'CryoSleep'], axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'CryoSleep'], axis=1, inplace=True)

columns_to_bool = ["VIP"]
train_df[columns_to_bool] = train_df[columns_to_bool].astype('bool')
test_df[columns_to_bool] = test_df[columns_to_bool].astype('bool')

def cabin_transfrom(df):
    for i in range(0 ,len(df)):
        if not(pd.isna(df.loc[i, 'Cabin'])):
            s = str(df.loc[i, 'Cabin'])
            df.loc[i, 'Cabin'] = s[0]
cabin_transfrom(train_df)
cabin_transfrom(test_df)

train_df.to_csv(os.path.join(data_path, "train.csv"))
test_df.to_csv(os.path.join(data_path, "test.csv"))

