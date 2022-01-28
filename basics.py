import pickle
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

model = pickle.load(open('m1.pkl', 'rb'))
dt1 = pd.read_csv('clustered_data.csv')
dt1.drop(['Unnamed: 0', 'race', 'engnat', 'hand', 'source'], axis=1, inplace=True)

scaled = minmax_scale(dt1.iloc[:, :-1])

# model = pickle.load(open('model.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(scaled, dt1['cluster'], test_size=0.33, random_state=5)

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

