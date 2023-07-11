import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import sklearn.utils as su
import sklearn.preprocessing as pp
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as metrics
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w

w.filterwarnings('ignore')

data = pd.read_csv("Data.csv")
choice = 0

while choice != 10:
    print("1. Marks Class Count Graph\n"
          "2. Marks Class Semester-wise Graph\n"
          "3. Marks Class Gender-wise Graph\n"
          "4. Marks Class Nationality-wise Graph\n"
          "5. Marks Class Grade-wise Graph\n"
          "6. Marks Class Section-wise Graph\n"
          "7. Marks Class Topic-wise Graph\n"
          "8. Marks Class Stage-wise Graph\n"
          "9. Marks Class Absent Days-wise Graph\n"
          "10. No Graph\n")
    choice = int(input("Enter Choice: "))

    if choice == 1:
        print("Loading Graph....\n")
        time.sleep(1)
        print("\tMarks Class Count Graph")
        axes = sns.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.show()

    elif choice == 2:
        print("Loading Graph....\n")
        time.sleep(1)
        print("\tMarks Class Semester-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 3:
        print("Loading Graph..\n")
        time.sleep(1)
        print("\tMarks Class Gender-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 4:
        print("Loading Graph..\n")
        time.sleep(1)
        print("\tMarks Class Nationality-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 5:
        print("Loading Graph: \n")
        time.sleep(1)
        print("\tMarks Class Grade-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09',
                                                                  'G-10', 'G-11', 'G-12'], hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 6:
        print("Loading Graph..\n")
        time.sleep(1)
        print("\tMarks Class Section-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 7:
        print("Loading Graph..\n")
        time.sleep(1)
        print("\tMarks Class Topic-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 8:
        print("Loading Graph..\n")
        time.sleep(1)
        print("\tMarks Class Stage-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

    elif choice == 9:
        print("Loading Graph..\n")
        time.sleep(1)
        print("\tMarks Class Absent Days-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
        plt.show()

if choice == 10:
    print("Exiting..\n")
    time.sleep(1)

# Remove unnecessary columns
data = data.drop(["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID", "Topic", "Semester",
                  "Relation", "ParentschoolSatisfaction", "ParentAnsweringSurvey", "AnnouncementsView"], axis=1)

# Shuffle the data
su.shuffle(data)

countD = 0
countP = 0
countL = 0
countR = 0
countN = 0

gradeID_dict = {
    "G-01": 1,
    "G-02": 2,
    "G-03": 3,
    "G-04": 4,
    "G-05": 5,
    "G-06": 6,
    "G-07": 7,
    "G-08": 8,
    "G-09": 9,
    "G-10": 10,
    "G-11": 11,
    "G-12": 12
}

data = data.replace({"GradeID": gradeID_dict})

for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]
feats_Train = feats[0:ind]
feats_Test = feats[(ind + 1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind + 1):len(lbls)]

modelD = st.DecisionTreeClassifier()
modelD.fit(feats_Train, lbls_Train)
lbls_predD = modelD.predict(feats_Test)

for a, b in zip(lbls_Test, lbls_predD):
    if a == b:
        countD += 1

accD = countD / len(lbls_Test)

print("\nAccuracy measures using Decision Tree:")
print(metrics.classification_report(lbls_Test, lbls_predD), "\n")
print("\nAccuracy using Decision Tree: ", str(round(accD, 3)))

time.sleep(1)

modelR = se.RandomForestClassifier()
modelR.fit(feats_Train, lbls_Train)
lbls_predR = modelR.predict(feats_Test)

for a, b in zip(lbls_Test, lbls_predR):
    if a == b:
        countR += 1

print("\nAccuracy Measures for Random Forest Classifier: \n")
print(metrics.classification_report(lbls_Test, lbls_predR), "\n")
accR = countR / len(lbls_Test)
print("\nAccuracy using Random Forest: ", str(round(accR, 3)))

time.sleep(1)

modelP = lm.Perceptron()
modelP.fit(feats_Train, lbls_Train)
lbls_predP = modelP.predict(feats_Test)

for a, b in zip(lbls_Test, lbls_predP):
    if a == b:
        countP += 1

accP = countP / len(lbls_Test)
print("\nAccuracy measures using Linear Model Perceptron:")
print(metrics.classification_report(lbls_Test, lbls_predP), "\n")
print("\nAccuracy using Linear Model Perceptron: ", str(round(accP, 3)), "\n")

time.sleep(1)

modelL = lm.LogisticRegression()
modelL.fit(feats_Train, lbls_Train)
lbls_predL = modelL.predict(feats_Test)

for a, b in zip(lbls_Test, lbls_predL):
    if a == b:
        countL += 1

accL = countL / len(lbls_Test)
print("\nAccuracy measures using Linear Model Logistic Regression:")
print(metrics.classification_report(lbls_Test, lbls_predL), "\n")
print("\nAccuracy using Linear Model Logistic Regression: ", str(round(accP, 3)), "\n")

time.sleep(1)

modelN = nn.MLPClassifier(activation="logistic")
modelN.fit(feats_Train, lbls_Train)
lbls_predN = modelN.predict(feats_Test)

for a, b in zip(lbls_Test, lbls_predN):
    if a == b:
        countN += 1

accN = countN / len(lbls_Test)
print("\nAccuracy measures using MLP Classifier:")
print(metrics.classification_report(lbls_Test, lbls_predN), "\n")
print("\nAccuracy using Neural Network MLP Classifier: ", str(round(accN, 3)), "\n")

choice = input("Do you want to test specific input (y or n): ")

if choice.lower() == "y":
    gen = input("Enter Gender (M or F): ")
    if gen.upper() == "M":
        gen = 1
    elif gen.upper() == "F":
        gen = 0

    nat = input("Enter Nationality: ")
    pob = input("Place of Birth: ")
    gra = input("Grade ID as (G-<grade>): ")
    if gra == "G-02":
        gra = 2
    elif gra == "G-04":
        gra = 4
    elif gra == "G-05":
        gra = 5
    elif gra == "G-06":
        gra = 6
    elif gra == "G-07":
        gra = 7
    elif gra == "G-08":
        gra = 8
    elif gra == "G-09":
        gra = 9
    elif gra == "G-10":
        gra = 10
    elif gra == "G-11":
        gra = 11
    elif gra == "G-12":
        gra = 12

    sec = input("Enter Section: ")
    top = input("Enter Topic: ")
    sem = input("Enter Semester (F or S): ")
    if sem.upper() == "F":
        sem = 0
    elif sem.upper() == "S":
        sem = 1

    rel = input("Enter Relation (Father or Mum): ")
    if rel == "Father":
        rel = 0
    elif rel == "Mum":
        rel = 1

    rai = int(input("Enter raised hands: "))
    res = int(input("Enter Visited Resources: "))
    ann = int(input("Enter announcements viewed: "))
    dis = int(input("Enter no. of Discussions: "))
    sur = input("Enter Parent Answered Survey (Y or N): ")
    if sur.upper() == "Y":
        sur = 1
    elif sur.upper() == "N":
        sur = 0

    sat = input("Enter Parent School Satisfaction (Good or Bad): ")
    if sat == "Good":
        sat = 1
    elif sat == "Bad":
        sat = 0

    absc = input("Enter No. of Absences (Under-7 or Above-7): ")
    if absc == "Under-7":
        absc = 1
    elif absc == "Above-7":
        absc = 0

    arr = np.array([rai, res, dis, absc])

    predD = modelD.predict(arr.reshape(1, -1))
    predR = modelR.predict(arr.reshape(1, -1))
    predP = modelP.predict(arr.reshape(1, -1))
    predL = modelL.predict(arr.reshape(1, -1))
    predN = modelN.predict(arr.reshape(1, -1))

    if predD == 0:
        predD = "H"
    elif predD == 1:
        predD = "M"
    elif predD == 2:
        predD = "L"

    if predR == 0:
        predR = "H"
    elif predR == 1:
        predR = "M"
    elif predR == 2:
        predR = "L"

    if predP == 0:
        predP = "H"
    elif predP == 1:
        predP = "M"
    elif predP == 2:
        predP = "L"

    if predL == 0:
        predL = "H"
    elif predL == 1:
        predL = "M"
    elif predL == 2:
        predL = "L"

    if predN == 0:
        predN = "H"
    elif predN == 1:
        predN = "M"
    elif predN == 2:
        predN = "L"

    time.sleep(1)
    print("\nUsing Decision Tree Classifier: ", predD)
    time.sleep(1)
    print("Using Random Forest Classifier: ", predR)
    time.sleep(1)
    print("Using Linear Model Perceptron: ", predP)
    time.sleep(1)
    print("Using Linear Model Logisitic Regression: ", predL)
    time.sleep(1)
    print("Using Neural Network MLP Classifier: ", predN)
    print("\nExiting...")
    time.sleep(1)

else:
    print("Exiting..")
    time.sleep(1)
