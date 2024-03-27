from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,disease_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Liver_Disease_Status(request):
    if request.method == "POST":

        Pid= request.POST.get('Pid')
        Age= request.POST.get('Age')
        Gender= request.POST.get('Gender')
        Total_Bilirubin= request.POST.get('Total_Bilirubin')
        Direct_Bilirubin= request.POST.get('Direct_Bilirubin')
        Alkaline_Phosphotase= request.POST.get('Alkaline_Phosphotase')
        Alamine_Aminotransferase= request.POST.get('Alamine_Aminotransferase')
        Aspartate_Aminotransferase= request.POST.get('Aspartate_Aminotransferase')
        Total_Protiens= request.POST.get('Total_Protiens')
        Albumin= request.POST.get('Albumin')
        Albumin_and_Globulin_Ratio= request.POST.get('Albumin_and_Globulin_Ratio')

        df = pd.read_csv('liver_patient.csv')
        df
        df.columns
        df.isnull().sum()


        def apply_results(results):
            if (results <= 1.2 and results>=0.1):
                return 0  # No Disease
            elif (results > 1.2):
                return 1  # Disease Found

        df['Results'] = df['Direct_Bilirubin'].apply(apply_results)

        cv = CountVectorizer()
        X = df['Pid']
        y = df['Results']

        print("PID")
        print(X)
        print("RESULTS")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)
        models.append(('LogisticRegression', reg))

        from sklearn.tree import DecisionTreeClassifier
        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        detection_accuracy.objects.create(names="Decision Tree Classifier",ratio=accuracy_score(y_test, dtcpredict) * 100)
        models.append(('DecisionTreeClassifier', dtc))

        print("Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        RFC = RandomForestClassifier(random_state=0)
        RFC.fit(X_train, y_train)
        pred_rfc = RFC.predict(X_test)
        RFC.score(X_test, y_test)
        print("ACCURACY")
        print(accuracy_score(y_test, pred_rfc) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, pred_rfc))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, pred_rfc))
        models.append(('RFC', RFC))

        print("KNeighborsClassifier")
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        knpredict = kn.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, knpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, knpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, knpredict))
        models.append(('KNeighborsClassifier', kn))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        patientid_data = [Pid]
        vector1 = cv.transform(patientid_data).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Liver Disease'
        elif prediction == 1:
            val = 'Foud Liver Disease'

        print(val)
        print(pred1)

        predicts = 'predicts.csv'
        df.to_csv(predicts, index=False)
        df.to_markdown

        print(val)
        print(pred1)

        disease_prediction.objects.create(Pid=Pid,Age=Age,Gender=Gender,Total_Bilirubin=Total_Bilirubin,Direct_Bilirubin=Direct_Bilirubin,Alkaline_Phosphotase=Alkaline_Phosphotase,Alamine_Aminotransferase=Alamine_Aminotransferase,Aspartate_Aminotransferase=Aspartate_Aminotransferase,Total_Protiens=Total_Protiens,Albumin=Albumin,Albumin_and_Globulin_Ratio=Albumin_and_Globulin_Ratio,prediction=val)

        return render(request, 'RUser/Predict_Liver_Disease_Status.html',{'objs': val})
    return render(request, 'RUser/Predict_Liver_Disease_Status.html')



