import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
boston_df = pd.read_csv('BostonHousing.csv')
print(boston_df.head())
print(boston_df.info())
print(boston_df.describe().round(2))
print(boston_df.isnull().sum())
corr_matrix = boston_df.corr().round(2)
print(corr_matrix)
x1 = boston_df['lstat']
x2 = boston_df['rm']
y = boston_df['medv']
plt.figure(figsize = (10,6))
plt.scatter(x1, y)

plt.xlabel('Процент населения с низким социальным статусом', fontsize = 15)
plt.ylabel('Медианная цена недвижимости, тыс. долларов', fontsize = 15)
plt.title('Социальный статус населения и цены на жилье', fontsize = 18)
plt.figure(figsize = (10,6))
plt.scatter(x2, y)

plt.xlabel('Среднее количество комнат', fontsize = 15)
plt.ylabel('Медианная цена недвижимости, тыс. долларов', fontsize = 15)
plt.title('Среднее количество комнат и цены на жилье', fontsize = 18)
X = boston_df[['rm', 'lstat', 'ptratio', 'tax', 'indus']]
y = boston_df['medv']
print(type(X), type(y))
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(y_pred[:5])

from sklearn import metrics

print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))
plt.show()