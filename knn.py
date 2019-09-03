import sklearn
import csv
import sys
import os
import constant
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics

# GLOBAL VARIABLES START
ctrlNumCounter = 0
globalCtrlNum = 0
globalExerNum = 1 # DAPAT DI NAGBABAGO ETO UNLESS MOVE ON NA SA NEXT EXERCISE.
globalcwNum = 0
globaldata = []
globalclass = []
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
    y = input("\nStart na natin machine learning? (enter \'y\' to start) ")
    if y in ['y', 'Y']:
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
    global globaldata, globalcwNum, globalclass

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
                    globalclass.append(False)
                else:
                    globalclass.append(True)
                globaldata.append(row)
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
        ctrlNumCounter += 1
    else:
        globalcwNum = 1

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
        for row in globaldata:
            for col in row:
                if col == "":
                    print("\nWait! A value is missing or is empty! Spotted after processing \'" + filename + "\'")
                    print("Chances are: it didn't record the values or the format is incorrect in the CSV")
                    print("Check the file mentioned above")
                    exit()

    while True:
        if increment_globals() == False:
            print("Data gathering finished.")
            break
        filename = generate_csv_filename()
        csv_to_array(filename)
        print("File \'" + filename + "\' processed.")
        for row in globaldata:
            for col in row:
                if col == "":
                    print("\nWait! A value is missing or is empty! Spotted after processing \'" + filename + "\'")
                    print("Chances are: it didn't record the values or the format is incorrect in the CSV")
                    print("Check the file mentioned above")
                    exit()
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def begin_machine_learning():
    global globaldata, globalclass

    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    globaldata = np.array(globaldata, dtype=np.float64)
    globalclass = np.array(globalclass, dtype=np.bool)

    print("\nSplitting for 60% training and 40% temp from 100% dataset... ", end="")
    X_train, X_tmp, y_train, y_tmp = train_test_split(globaldata, globalclass, test_size = 1 - train_ratio)
    print("Done.")
    print("Splitting for 20% validation and 20% testing from 40% temp... ", end="")
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size = test_ratio / (test_ratio + validation_ratio))
    print("Done.")

    countTrain = 0
    for out in y_train:
        countTrain += 1
    print("\nTraining Dataset: " + str(int(countTrain)) + " observations")
    countVal = 0
    for out in y_val:
        countVal += 1
    print("Validation Dataset: " + str(int(countVal)) + " observations")
    countTest = 0
    for out in y_test:
        countTest += 1
    print("Test Dataset: " + str(int(countTest)) + " observations")
    print("Total Number of Observations: " + str(countTrain + countVal + countTest) + "")

    print("\nTRAINING THE MODEL w/ KNN USING Training Dataset... ", end="")
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, y_train)
    print("Done.")

    print("\nVALIDATING THE MODEL USING THE Validation Dataset...")
    c = input("Given that there are " + str(int(countVal)) + " observations, how many folds will there be? (Default: 10) ")
    if str(c).isnumeric():
        folderNum = int(c)
    else:
        exit()

    obsVal = countVal / folderNum
    print("\nThere are currently " + str(countVal) + " unique observations in the Validation Dataset.")
    print("You decided to apply a " + str(int(folderNum)) + "-Fold Cross-Validation.")
    print("Thus, each fold/group will have " + str(obsVal) + " observations.")
    d = input("\nStart validation phase? (enter \'y\' to start) ")
    if d != "y":
        exit()
    # kfold
    print("Applying " + str(folderNum) + "-Fold Cross-Validation... ")
    accuracySum = 0
    kfold = KFold(folderNum, True, 1)
    for train_index, test_index in kfold.split(X_val):
        X_valtrain, X_valtest = X_val[train_index], X_val[test_index]
        y_valtrain, y_valtest = y_val[train_index], y_val[test_index]
        print("Validating iteration...", end="")
        ml = KNeighborsClassifier(n_neighbors = 3)
        ml.fit(X_valtrain, y_valtrain)
        y_pred = ml.predict(X_valtest)
        res = metrics.accuracy_score(y_valtest, y_pred)
        accuracySum += (res * 100)
        print(" Score: " + str(res))

    print("\n\nAverage Accuracy during Validation Phase: " + str(accuracySum / folderNum))
    print("\nHave you written down the OTHER algorithm's average accuracy? If not, write down this")
    print("program's results (the one above), and STOP this program immediately. You will have")
    print("to execute the other program to get its Average Accuracy, and write that down.")
    print("\nIf you have written down both programs\' average accuracy, make sure you're currently")
    f = input("executing the program that has a higher average accuracy. If so, enter 'y' to continue.\n")
    if f != "y":
        exit()

    print("\nEVALUATING FOR CONFUSION MATRIX SCORE... ")
    y_pred = knn.predict(X_test)
    res = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy Score:", res)
    print("Report")
    print(metrics.classification_report(y_test, y_pred))

    print("yay")
# -----------------------------------------------------------------------------------------
#OTHER FUNCTIONS END

gogogo()
