import sklearn
import csv
import sys
import os
import constant
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics

# GLOBAL VARIABLES START
ctrlNumCounter = 0
globalCtrlNum = 0
globalExerNum = 1 # DAPAT DI NAGBABAGO ETO UNLESS MOVE ON NA SA NEXT EXERCISE.
globalcwNum = 0
globaldata_A = []
globaldata_B = []
globaldata_C = []
globaldata_D = []
globaldata_E = []
globalclass_A = []
globalclass_B = []
globalclass_C = []
globalclass_D = []
globalclass_E = []
sampleSize = 0
# GLOBAL VARIABLES END

# MAIN FUNCTION START
# -----------------------------------------------------------------------------------------
def gogogo():
    global ctrlNumCounter, globalCtrlNum, globalExerNum, globalcwNum, sampleSize

    print("Running " + os.path.basename(sys.argv[0]) + "\n")
    print("Scikit-Learn version: " + sklearn.__version__)
    print("Executing Python script...\n")

    n = input("Hanggang saang Control Number ang babasahin ng program? (lowest is 0) ")
    sampleSize = int(n)
    gather_datasets()
    begin_machine_learning()
# -----------------------------------------------------------------------------------------
# MAIN FUNCTION END

# OTHER FUNCTIONS START
# -----------------------------------------------------------------------------------------
def generate_csv_filename():
    global ctrlNumCounter, globalCtrlNum, globalExerNum, globalExerNum, sampleSize

    result = ""
    if ctrlNumCounter > 9:
        result = "person-" + str(ctrlNumCounter) + "-" + str(globalExerNum) + "-" + str(globalcwNum) + ".csv"
    else:
        result = "person-0" + str(ctrlNumCounter) + "-" + str(globalExerNum) + "-" + str(globalcwNum) + ".csv"

    return result
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def csv_to_array(name):
    global globaldata_A, globaldata_B, globaldata_C, globaldata_D, globaldata_E, globalclass_A, globalclass_B, globalclass_C, globalclass_D, globalclass_E, globalcwNum

    file = constant.DATA_PATH + "\\" + name
    rows = []
    try:
        with open(file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            counter = 0
            for row in csv_reader:
                if(line_count % 2 == 0):
                    line_count += 1
                    continue
                else:
                    rows.append(row)
                    line_count += 1
                    counter += 1
            for row in rows:
                if globalcwNum == 0:
                    if globalExerNum == 1:
                        globalclass_A.append(False)
                    elif globalExerNum == 2:
                        globalclass_B.append(False)
                    elif globalExerNum == 3:
                        globalclass_C.append(False)
                    elif globalExerNum == 4:
                        globalclass_D.append(False)
                    elif globalExerNum == 5:
                        globalclass_E.append(False)
                else:
                    if globalExerNum == 1:
                        globalclass_A.append(True)
                    elif globalExerNum == 2:
                        globalclass_B.append(True)
                    elif globalExerNum == 3:
                        globalclass_C.append(True)
                    elif globalExerNum == 4:
                        globalclass_D.append(True)
                    elif globalExerNum == 5:
                        globalclass_E.append(True)
                if globalExerNum == 1:
                    globaldata_A.append(row)
                elif globalExerNum == 2:
                    globaldata_B.append(row)
                elif globalExerNum == 3:
                    globaldata_C.append(row)
                elif globalExerNum == 4:
                    globaldata_D.append(row)
                elif globalExerNum == 5:
                    globaldata_E.append(row)
    except:
        print("Exception Error occured. File \'" + name + "\' not found(?).")
        exit()

    return True
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def increment_globals():
    global ctrlNumCounter, globalCtrlNum, globalExerNum, globalExerNum, globalcwNum, sampleSize

    if globalcwNum == 1:
        globalcwNum = 0
        globalExerNum += 1
    else:
        globalcwNum = 1

    if globalExerNum > 5:
        globalExerNum = 1
        ctrlNumCounter += 1

    if ctrlNumCounter > sampleSize:
        print("\nLast file has been reached.")
        return False

    return True
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def gather_datasets():
    global ctrlNumCounter, globalCtrlNum, globalExerNum, globalcwNum

    filename = ""

    print("\nOpening CSV files: ")
    if ctrlNumCounter == 0:
        filename = generate_csv_filename()
        csv_to_array(filename)
        print("File \'" + filename + "\' processed.")

    while True:
        if increment_globals() == False:
            print("Data gathering finished.")
            break
        filename = generate_csv_filename()
        csv_to_array(filename)
        print("File \'" + filename + "\' processed.")
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def begin_machine_learning():
    global globaldata_A, globaldata_B, globaldata_C, globaldata_D, globaldata_E, globalclass_A, globalclass_B, globalclass_C, globalclass_D, globalclass_E

    train_ratio = 0.70
    validation_ratio = 0.20
    test_ratio = 0.10
    globaldata_A = np.array(globaldata_A, dtype=np.float64)
    globaldata_B = np.array(globaldata_B, dtype=np.float64)
    globaldata_C = np.array(globaldata_C, dtype=np.float64)
    globaldata_D = np.array(globaldata_D, dtype=np.float64)
    globaldata_E = np.array(globaldata_E, dtype=np.float64)
    globalclass_A = np.array(globalclass_A, dtype=np.bool)
    globalclass_B = np.array(globalclass_B, dtype=np.bool)
    globalclass_C = np.array(globalclass_C, dtype=np.bool)
    globalclass_D = np.array(globalclass_D, dtype=np.bool)
    globalclass_E = np.array(globalclass_E, dtype=np.bool)

    alldata = [globaldata_A, globaldata_B, globaldata_C, globaldata_D, globaldata_E]
    allclass = [globalclass_A, globalclass_B, globalclass_C, globalclass_D, globalclass_E]

    lastpred = []
    lasttest = []

    for i in range(5):
        X_train, X_tmp, y_train, y_tmp = train_test_split(alldata[i], allclass[i], test_size = 1 - train_ratio)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size = test_ratio / (test_ratio + validation_ratio))
        # knn = KNeighborsClassifier(n_neighbors = 3)
        # knn.fit(X_train, y_train)
        ert = ExtraTreesClassifier(n_estimators = 64, random_state = 0)
        ert.fit(X_train, y_train)
        print("Testing for Set " + str(i) + "... ", end="")
        y_pred = ert.predict(X_test)
        # y_pred = knn.predict(X_test)
        for x in y_pred:
            lastpred.append(x)
        for y in y_test:
            lasttest.append(y)
        print("Done.")

    # countTrain = 0
    # for out in y_train:
    #     countTrain += 1
    # print("\nTraining Dataset: " + str(int(countTrain)) + " observations")
    # countVal = 0
    # for out in y_val:
    #     countVal += 1
    # print("Validation Dataset: " + str(int(countVal)) + " observations")
    # countTest = 0
    # for out in y_test:
    #     countTest += 1
    # print("Test Dataset: " + str(int(countTest)) + " observations")
    # print("Total Number of Observations: " + str(countTrain + countVal + countTest) + "")


    print("\nEVALUATING FOR CONFUSION MATRIX SCORE... ")

    print("Accuracy Score:", metrics.accuracy_score(lasttest, lastpred))
    print("Precision Score:", metrics.precision_score(lasttest, lastpred))
    print("Recall:", metrics.recall_score(lasttest, lastpred))
    print("F1-Score:", metrics.f1_score(lasttest,lastpred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(lasttest, lastpred))
    print(metrics.classification_report(lasttest, lastpred))
# -----------------------------------------------------------------------------------------
#OTHER FUNCTIONS END

gogogo()
