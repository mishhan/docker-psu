from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

# Reading (localhost is changed to db) 
con_string = 'postgresql://postgres@db:5432/data_db'
engine = create_engine(con_string)
data = pd.read_sql_query('select * from "bg_statistic"',con=engine)

print("First five rows:")
print(data.head())
print('Size: {0} rows {1} columns..'.format(data.shape[0], data.shape[1]))

# Preprocessing

# Win = !Lose
data['Result'] = data['Win'] == 1
data['Result'] = data['Result'].astype(int)
data = data.drop(columns=['Win', 'Lose'])
data.head()
print('---------------------------------------------')

# drop useless columns
data = data.drop(columns=['Code'])
data = data.drop(columns=['BE'])
data = data.drop(columns=["Role"])
data.head()
print('---------------------------------------------')

# print whole data infromation
data.info()
print('---------------------------------------------')

# take a look at data
data_bar = data.groupby(['Faction','Result']).size().unstack(fill_value=0)
axes = data_bar.plot.bar()
axes.set_xlabel("Faction")
axes.set_ylabel("Count win/loss")
plt.show(block=True)

data_bar = data.groupby(['Result','Faction','Battleground']).size().unstack(fill_value=0)
axes = data_bar.plot.bar(figsize=(20, 5))
axes.set_xlabel("Faction in Battleground")
axes.set_ylabel("Count win/loss")
plt.show(block=True)

data_bar = data.groupby(['Faction','Result','Class']).size().unstack(fill_value=0)
axes = data_bar.plot.bar(figsize=(20, 5))
axes.set_xlabel("Class in Faction")
axes.set_ylabel("Count win/loss")
plt.show(block=True)

# ...

# convert categorial into numeric
numeric_columns = ['KB', 'D', 'HK', 'DD', 'HD', 'Honor']
result_column = ['Result']
categorial_columns = list(set(data.columns.values.tolist()) - set(numeric_columns) - set(result_column))
print("Numeric columns")
print(numeric_columns)
print("Categorial columns")
print(categorial_columns)
print("Result column")
print(result_column)
print(data[categorial_columns])
print('---------------------------------------------')

labelencoder = preprocessing.LabelEncoder()
for col in categorial_columns:
    data[col] = labelencoder.fit_transform(data[col])
print(data[categorial_columns])
print('---------------------------------------------')

print("Correlation:")
# count correlation
cor_columns = numeric_columns + categorial_columns
print(data[cor_columns].corrwith(data[result_column[0]]))
print('---------------------------------------------')

# split data on train and test sets
from sklearn.model_selection import train_test_split
x = data[cor_columns]
y = data[result_column[0]]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
lg = LogisticRegression()
lg.fit(x, y)
for i in zip(data.columns, lg.coef_):
    print(i)
y_prob = lg.predict(x_test)
y_pred = np.where(y_prob > 0.5, 1, 0)
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_pred))
print('---------------------------------------------')
print('Accuracy (using matrixes):')
matrix = metrics.confusion_matrix(y_test,y_pred)
sum_diag = matrix.diagonal().sum()
sum_full = matrix.sum()
print(pd.crosstab(y_test, y_pred, rownames=['TRUE'], colnames=['RESULT']))
print('---------------------------------------------')

print("END")

