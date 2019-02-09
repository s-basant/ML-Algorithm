import pandas as pd
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("DataSet-Release 1/ds2/ds2Train.csv", header=None)
validation = pd.read_csv("DataSet-Release 1/ds2/ds2Val.csv", header=None)

y = data.iloc[:, -1]
x = data.iloc[:, :-1]

vx = validation.iloc[:, :-1]
vy = validation.iloc[:, -1]

clf = svm.SVC(C = 10.0, gamma = 0.01, random_state= 1)
clf.fit(x, y)


vp = clf.predict(vx)



print(accuracy_score(vy, vp))
rp = classification_report(vy,vp )
print(rp)



count =0
file = open('Output/ds2Val-4.csv', 'w')

for x in vp:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()