
import pandas as pd

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#gnb = GaussianNB()
gnb = GaussianNB(var_smoothing=.07)

data = pd.read_csv("DataSet-Release 1/ds1/ds1Train.csv", header=None)
validation = pd.read_csv("DataSet-Release 1/ds1/ds1Val.csv", header=None)

y = data.iloc[:, -1]
x = data.iloc[:, :-1]

vx = validation.iloc[:, :-1]
vy = validation.iloc[:, -1]

#y_pred = gnb.fit(x, y).predict(vx)

clf = gnb.fit(x, y)
vp = clf.predict(vx)



print(accuracy_score(vy, vp))
rp = classification_report(vy,vp )
print(rp)



count =0
file = open('Output/ds1Val-nb.csv', 'w')

for x in vp:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()