The folder contains 4 items.
1> binaryFile
2> Mini Project 2 Report.pdf
3> Output
4> PythonScript

1> binaryFile : The folder contain 8 binary files and a driver file
    dt_ds1.sav => Classifier : Decision Tree ; Trained on : Dataset 1
    dt_ds2.sav => Classifier : Decision Tree ; Trained on : Dataset 2
    nb_ds1.sav => Classifier : Naive Bayes ; Trained on : Dataset 1
    nb_ds2.sav => Classifier : Naive Bayes ; Trained on : Dataset 2
    rf_ds1.sav => Classifier : Random Forest ; Trained on : Dataset 1
    rf_ds2.sav => Classifier : Random Forest ; Trained on : Dataset 2
    svm_ds1.sav => Classifier : SVM ; Trained on : Dataset 1
    svm_ds2.sav => Classifier : SVM ; Trained on : Dataset 2

    driver.py : This is driver python scrip using which above listed model can be excuted.
    Once the script is executed, user will be asked to chcoose a classifier and than a dataset. Following which the output file will be created as required.

    Before executing the driver.py , make sure that the test file has been kept at below mentioned path.
    test file for dataset 1 =>  'DataSet-Release 2/ds1/ds1Test.csv'
    test file for dataset 2=> 'DataSet-Release 2/ds2/ds2Test.csv'
    and
    output file for dataset1 and for classifier 'xx' are stored  => 'Output/ds1Test-xx.csv
    output file for dataset2 and for classifier 'xx' are stored  => 'Output/ds2Test-xx.csv

2> Mini Project 2 Report.pdf => Report on analysis and experimentation for 4 type of classifiers

3> Output > It has 16 output file
            For 4 different classifers each has
            for xx classifier,  1 for test on dataset 1 => ds1Test-xx.csv
            for xx classifier,  1 for test on dataset 2 => ds2Test-xx.csv
            for xx classifier,  1 for validation on dataset 1 => ds1Val-xx.csv
            for xx classifier,  1 for validation on dataset 2 => ds2Val-xx.csv
            xx = [decision tree, Random Forest, Naive Bayes, SVM]

4> Python Script: It contains all script used during experimentaiton and generation of binary file