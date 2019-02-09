import pandas as pd
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.externals import joblib


# save the model to disk
filename = 'svm_ds1.sav'

data = pd.read_csv("DataSet-Release 2/ds1/ds1Train.csv", header=None)
test = pd.read_csv("DataSet-Release 2/ds1/ds1Test.csv", header=None)

y = data.iloc[:, -1]
x = data.iloc[:, :-1]

vx = test.iloc[:, :]

clf = svm.SVC(C = 10.0, gamma = 0.001, random_state= 1)
clf.fit(x, y)

joblib.dump(clf, filename)

vp = clf.predict(vx)


count =0
file = open('Output/ds1Test-4.csv', 'w')

for x in vp:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()