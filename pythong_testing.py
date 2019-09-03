#wla lang. pangtest lang
import constant
import csv

# GLOBAL VARIABLES START
ctrlNumCounter = 0
globalCtrlNum = 0
globalExerNum = 1 # DAPAT DI NAGBABAGO ETO.
globalcwNum = 0
sampleSize = 1
# GLOBAL VARIABLES END


def increment_globals():
    global ctrlNumCounter, globalCtrlNum, globalExerNum, globalExerNum, globalcwNum, sampleSize

    if ctrlNumCounter > sampleSize:
        print("Last file has been reached")
        return False

    if globalcwNum == 1:
        globalcwNum = 0
        ctrlNumCounter += 1
    else:
        globalcwNum = 1


    generate_csv_filename()
    return True

def generate_csv_filename():
    global ctrlNumCounter, globalCtrlNum, globalExerNum, globalExerNum, sampleSize

    filename = ""

    if ctrlNumCounter > 9:
        filename = "person-" + str(ctrlNumCounter) + "-" + str(globalExerNum) + "-" + str(globalcwNum) + ".csv"
    else:
        filename = "person-0" + str(ctrlNumCounter) + "-" + str(globalExerNum) + "-" + str(globalcwNum) + ".csv"

    print(filename)

while True:
    if increment_globals() == False:
        break


#
# globaldata = []
# globalclass = []
# globalcwNum = 0
#
# def csv_to_array(name):
#
#     global globaldata, globalcwNum, globalclass
#
#     file = constant.DATA_PATH_TEST + "\\" + filename
#     rows = []
#
#     with open(file, mode='r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         line_count = 0
#         counter = 0
#         for row in csv_reader:
#             if(line_count % 2 == 0):
#                 line_count += 1
#                 continue
#             else:
#                 rows.append(row)
#                 line_count += 1
#                 counter += 1
#         for row in rows:
#             if globalcwNum == 0:
#                 globalclass.append("wrong")
#             else:
#                 globalclass.append("correct")
#             globaldata.append(row)
#
# csv_to_array(filename)
