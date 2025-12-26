# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

df_clean = df.dropna(subset=['status'])

x = df_clean [['MDVP:Jitter(%)', 'MDVP:Fo(Hz)']]
y = df_clean[['status']]


sns.pairplot(df_clean, vars=['MDVP:Fo(Hz)', 'MDVP:Jitter(%)'], hue= 'status')
plt.suptitle('Pair Plot of Numrical Features by Status', y=1.02)
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scales = scaler.fit_transform(x)
len(x_scales)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scales, y, test_size=0.2)
len(x_train), len(x_test)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy}")

