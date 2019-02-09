
#########################################################
#                                                       #
#   This is Decision tree classifier using Dataset 2    #
#                                                       #
##########################################################


import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report







data = pd.read_csv("DataSet-Release 1/ds2/ds2Train.csv", header=None)
validation = pd.read_csv("DataSet-Release 1/ds2/ds2Val.csv", header=None)

y = data.iloc[:, -1]
x = data.iloc[:, :-1]

vx = validation.iloc[:, :-1]
vy = validation.iloc[:, -1]


# best parameter finding

# parameters = {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],  'max_features': ['auto', 'sqrt'],
#              'min_samples_leaf': [1, 2, 4],
#              'min_samples_split': [2, 5, 10],
#              'criterion': ['gini', 'entropy'],
#              }

# dt_clsr = tree.DecisionTreeClassifier(random_state = 4)
# dt_clf= GridSearchCV(dt_clsr, parameters , cv= 5)
# dt_clf.fit(x,y)
# dt_clf.best_params_


# trained the claissifer with parameter which were obtained from grid search
#
# classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=30,max_features='auto',
#                                           min_samples_leaf=1, min_samples_split=2, random_state=1)
classifier = tree.DecisionTreeClassifier(random_state=1, max_depth=30)

# classifier = tree.DecisionTreeClassifier(random_state=1)

def prune(decisiontree, min_samples_split=1):
    if decisiontree.min_samples_split >= min_samples_split:
        raise Exception('Tree already more pruned')
    else:
        decisiontree.min_samples_split = min_samples_split
        tree = decisiontree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_split:
                tree.children_left[i] = -1
                tree.children_right[i] = -1


clf = classifier.fit(x, y)
#prune(clf, min_samples_split=6)



vp = clf.predict(vx)
print(accuracy_score(vy, vp))
rp = classification_report(vy,vp )
print(rp)



count =0
file = open('Output/ds2Val-dt.csv', 'w')

for x in vp:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()