#########################################################
#                                                       #
#   This is Decision tree classifier using Dataset 1    #
#                                                       #
##########################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.externals import joblib


# save the model to disk
filename = 'rf_ds1.sav'

data = pd.read_csv("DataSet-Release 2/ds1/ds1Train.csv", header=None)
test = pd.read_csv("DataSet-Release 2/ds1/ds1Test.csv", header=None)

#data = pd.read_csv("ds2/ds2Train.csv", header=None)
#validation = pd.read_csv("ds2/ds2Val.csv", header=None)

y = data.iloc[:, -1]
x = data.iloc[:, :-1]

vp = test.iloc[:, :]
# trained the claissifer with parameter which were obtained from grid search
# classifier = RandomForestClassifier(random_state=2)
# classifier = RandomForestClassifier(random_state=2,criterion= 'entropy', max_depth=20,
#                                     min_samples_leaf=1,n_estimators = 200,min_samples_split=5)
# classifier = RandomForestClassifier(criterion='entropy', max_depth=60, min_samples_leaf=1,n_estimators = 700, min_samples_split=2, random_state=2, max_features= 'auto')
classifier = RandomForestClassifier(random_state=1, max_depth=20, n_estimators = 400)

clf = classifier.fit(x, y)

joblib.dump(clf, filename)

count =0
file = open('Output/ds1Test-3.csv', 'w')

for x in vp:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()