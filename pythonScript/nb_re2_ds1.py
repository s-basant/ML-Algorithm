
import pandas as pd

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.externals import joblib


# save the model to disk
filename = 'nb_ds1.sav'

data = pd.read_csv("DataSet-Release 2/ds1/ds1Train.csv", header=None)
test = pd.read_csv("DataSet-Release 2/ds1/ds1Test.csv", header=None)

y = data.iloc[:, -1]
x = data.iloc[:, :-1]

vx = test.iloc[:, :]

gnb = GaussianNB(var_smoothing=.07)

clf = gnb.fit(x, y)
joblib.dump(clf, filename)


vp = clf.predict(vx)




count =0
file = open('Output/ds1Test-nb.csv', 'w')

for x in vp:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()