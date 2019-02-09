
import imp
import pandas as pd
from sklearn.externals import joblib


print("Select classifier from below listed option. \n"
      "Enter '1' => Decision Tree \n"
      "Enter '2' => Naive Bayes \n"
      "Enter '3' => SVM \n"
      "Enter '4' => Random Forest \n")
classifierType = int(input("Enter the options mentioned above to select a classifier \n"))

print("Select dataset from below listed option. \n"
      "Enter '1' => Dataset 1 \n"
      "Enter '2' => Dataset 2 \n")
dataset = int(input("Enter the options mentioned above to select a dataset \n"))


if dataset == 1:
    test = pd.read_csv("DataSet-Release 2/ds1/ds1Test.csv", header=None)
    if classifierType == 1:
        filename = 'dt_ds1.sav'  # model built from
        outputFilePath = 'Output/ds1Test-dt.csv'
    elif classifierType == 2:
        filename = 'nb_ds1.sav'
        outputFilePath = 'Output/ds1Test-nb.csv'
    elif classifierType == 3:
        filename = 'svm_ds1.sav'
        outputFilePath = 'Output/ds1Test-svm.csv'
    elif classifierType == 4:
        filename = 'rf_ds1.sav'
        outputFilePath = 'Output/ds1Test-rf.csv'
    else:
        print(classifierType + " is not a valid Classifier type" )

elif dataset == 2:
    test = pd.read_csv("DataSet-Release 2/ds2/ds2Test.csv", header=None)
    if classifierType == 1:
        filename = 'dt_ds2.sav'
        outputFilePath = 'Output/ds2Test-dt.csv'
    elif classifierType == 2:
        filename = 'nb_ds2.sav'
        outputFilePath = 'Output/ds2Test-nb.csv'
    elif classifierType == 3:
        filename = 'svm_ds2.sav'
        outputFilePath = 'Output/ds2Test-svm.csv'
    elif classifierType == 4:
        filename = 'rf_ds2.sav'
        outputFilePath = 'Output/ds2Test-rf.csv'
    else:
        print(classifierType + " is not a valid Classifier type")
else:
    print(dataset + " is not a valid dataset")

x = test.iloc[:, :]


loaded_model_clf = joblib.load(filename)

p = loaded_model_clf.predict(x)

file = open(outputFilePath, 'w')

count =0
for x in p:
    count=count +1
    file.write(str(count))
    file.write(', ')
    file.write(str(x))
    file.write('\n')
file.close()