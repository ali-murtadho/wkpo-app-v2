from .models import ClassificationResult
from django.http import HttpResponse
from django.shortcuts import render, render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix ,accuracy_score, recall_score, precision_score, f1_score, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import cross_val_score
from django.utils.encoding import smart_str
from django.contrib.auth.decorators import login_required

import logging
import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64

logger = logging.getLogger(__name__)
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    error_message = None
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"You are now logged in as {username}.")
                return redirect('home')
            else:
                error_message = "Invalid username or password."
                messages.error(request, error_message)
        else:
            error_message = "Invalid username or password."
            messages.error(request, error_message)
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form, 'error_message': error_message})

def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.") 
    return redirect('index')


# Parameter grid for DecisionTreeClassifier
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

class Preprocessing_read_csv:
    def read_data(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\padi-clean.csv')
        df = pd.read_csv(file_path)
        return df
    def data_train_real_three(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\data_latih_real_03.csv')
        df = pd.read_csv(file_path)
        return df
    def data_test_real_three(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\data_uji_real_03.csv')
        df = pd.read_csv(file_path)
        return df
    def data_train_model_three(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\data_latih_model_03.csv')
        df = pd.read_csv(file_path)
        return df
    def data_test_model_three(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\data_uji_model_03.csv')
        df = pd.read_csv(file_path)
        return df
    def x_train_r_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_train_r_03.csv')
        df = pd.read_csv(file_path)
        return df
    def x_test_r_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_test_r_03.csv')
        df = pd.read_csv(file_path)
        return df
    def y_train_r_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_train_r_03.csv')
        df = pd.read_csv(file_path)
        return df
    def y_test_r_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_test_r_03.csv')
        df = pd.read_csv(file_path)
        return df
    def x_train_r_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_train_r_04.csv')
        df = pd.read_csv(file_path)
        return df
    def x_test_r_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_test_r_04.csv')
        df = pd.read_csv(file_path)
        return df
    def y_train_r_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_train_r_04.csv')
        df = pd.read_csv(file_path)
        return df
    def y_test_r_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_test_r_04.csv')
        df = pd.read_csv(file_path)
        return df
    def x_train_r_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_train_r_05.csv')
        df = pd.read_csv(file_path)
        return df
    def x_test_r_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_test_r_05.csv')
        df = pd.read_csv(file_path)
        return df
    def y_train_r_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_train_r_05.csv')
        df = pd.read_csv(file_path)
        return df
    def y_test_r_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_test_r_05.csv')
        df = pd.read_csv(file_path)
        return df
    def x_train_m_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_train_m_03.csv')
        df = pd.read_csv(file_path)
        return df
    def x_test_m_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_test_m_03.csv')
        df = pd.read_csv(file_path)
        return df
    def y_train_m_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_train_m_03.csv')
        df = pd.read_csv(file_path)
        return df
    def y_test_m_03(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_test_m_03.csv')
        df = pd.read_csv(file_path)
        return df
    def x_train_m_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_train_m_04.csv')
        df = pd.read_csv(file_path)
        return df
    def x_test_m_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_test_m_04.csv')
        df = pd.read_csv(file_path)
        return df
    def y_train_m_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_train_m_04.csv')
        df = pd.read_csv(file_path)
        return df
    def y_test_m_04(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_test_m_04.csv')
        df = pd.read_csv(file_path)
        return df
    def x_train_m_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_train_m_05.csv')
        df = pd.read_csv(file_path)
        return df
    def x_test_m_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\X_test_m_05.csv')
        df = pd.read_csv(file_path)
        return df
    def y_train_m_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_train_m_05.csv')
        df = pd.read_csv(file_path)
        return df
    def y_test_m_05(self):
        file_path = os.path.join(settings.BASE_DIR, 'model\data\y_test_m_05.csv')
        df = pd.read_csv(file_path)
        return df
    
@login_required(login_url='login')
def training(request):
    x_train_r_03 = Preprocessing_read_csv().x_train_r_03()
    y_train_r_03 = Preprocessing_read_csv().y_train_r_03()
    x_train_r_04 = Preprocessing_read_csv().x_train_r_04()
    y_train_r_04 = Preprocessing_read_csv().y_train_r_04()
    x_train_r_05 = Preprocessing_read_csv().x_train_r_05()
    y_train_r_05 = Preprocessing_read_csv().y_train_r_05()

    x_train_m_03 = Preprocessing_read_csv().x_train_m_03()
    y_train_m_03 = Preprocessing_read_csv().y_train_m_03()
    x_train_m_04 = Preprocessing_read_csv().x_train_m_04()
    y_train_m_04 = Preprocessing_read_csv().y_train_m_04()
    x_train_m_05 = Preprocessing_read_csv().x_train_m_05()
    y_train_m_05 = Preprocessing_read_csv().y_train_m_05()

    if 'train' in request.POST:
        best_model_m03 = DecisionTreeClassifier(max_depth=8).fit(x_train_m_03, y_train_m_03)
        best_model_m04 = DecisionTreeClassifier(max_depth=8).fit(x_train_m_04, y_train_m_04)
        best_model_m05 = DecisionTreeClassifier(max_depth=8).fit(x_train_m_05, y_train_m_05)

        best_model_r03 = DecisionTreeClassifier(max_depth=8).fit(x_train_r_03, y_train_r_03)
        best_model_r04 = DecisionTreeClassifier(max_depth=8).fit(x_train_r_04, y_train_r_04)
        best_model_r05 = DecisionTreeClassifier(max_depth=8).fit(x_train_r_05, y_train_r_05)
        
        pickle.dump(best_model_r03, open('model_r3.pkl', 'wb'))
        pickle.dump(best_model_r04, open('model_r4.pkl', 'wb'))
        pickle.dump(best_model_r05, open('model_r5.pkl', 'wb'))
        pickle.dump(best_model_m03, open('model_m3.pkl', 'wb'))
        pickle.dump(best_model_m04, open('model_m4.pkl', 'wb'))
        pickle.dump(best_model_m05, open('model_m5.pkl', 'wb'))

        title = "Training Page"
        messages.success(request, 'Model berhasil di pickle!')
        return render(request, 'result.html', {
            'title' : title,
            'success_message' : 'Data berhasil dilatih dan berhasil membuat model!'
        })
    
@login_required(login_url='login')
def testing(request):
    if 'test' in request.POST:
        model_r03 = pickle.load(open('model_r3.pkl', 'rb'))
        model_r04 = pickle.load(open('model_r4.pkl', 'rb'))
        model_r05 = pickle.load(open('model_r5.pkl', 'rb'))

        model_m03 = pickle.load(open('model_m3.pkl', 'rb'))
        model_m04 = pickle.load(open('model_m4.pkl', 'rb'))
        model_m05 = pickle.load(open('model_m5.pkl', 'rb'))

        y_pred_r03 = model_r03.predict(Preprocessing_read_csv().x_test_r_03())
        y_pred_r04 = model_r04.predict(Preprocessing_read_csv().x_test_r_04())
        y_pred_r05 = model_r05.predict(Preprocessing_read_csv().x_test_r_05())
        y_pred_m03 = model_m03.predict(Preprocessing_read_csv().x_test_m_03())
        y_pred_m04 = model_m04.predict(Preprocessing_read_csv().x_test_m_04())
        y_pred_m05 = model_m05.predict(Preprocessing_read_csv().x_test_m_05())

        y_pred_train_r03 = model_r03.predict(Preprocessing_read_csv().x_train_r_03())
        y_pred_train_r04 = model_r04.predict(Preprocessing_read_csv().x_train_r_04())
        y_pred_train_r05 = model_r05.predict(Preprocessing_read_csv().x_train_r_05())
        y_pred_train_m03 = model_m03.predict(Preprocessing_read_csv().x_train_m_03())
        y_pred_train_m04 = model_m04.predict(Preprocessing_read_csv().x_train_m_04())
        y_pred_train_m05 = model_m05.predict(Preprocessing_read_csv().x_train_m_05())

        acc_r03 = accuracy_score(Preprocessing_read_csv().y_test_r_03(), y_pred_r03)
        acc_r04 = accuracy_score(Preprocessing_read_csv().y_test_r_04(), y_pred_r04)
        acc_r05 = accuracy_score(Preprocessing_read_csv().y_test_r_05(), y_pred_r05)
        acc_m03 = accuracy_score(Preprocessing_read_csv().y_test_m_03(), y_pred_m03)
        acc_m04 = accuracy_score(Preprocessing_read_csv().y_test_m_04(), y_pred_m04)
        acc_m05 = accuracy_score(Preprocessing_read_csv().y_test_m_05(), y_pred_m05)

        rec_r03 = recall_score(Preprocessing_read_csv().y_test_r_03(), y_pred_r03, average='weighted')
        rec_r04 = recall_score(Preprocessing_read_csv().y_test_r_04(), y_pred_r04, average='weighted')
        rec_r05 = recall_score(Preprocessing_read_csv().y_test_r_05(), y_pred_r05, average='weighted')
        rec_m03 = recall_score(Preprocessing_read_csv().y_test_m_03(), y_pred_m03, average='weighted')
        rec_m04 = recall_score(Preprocessing_read_csv().y_test_m_04(), y_pred_m04, average='weighted')
        rec_m05 = recall_score(Preprocessing_read_csv().y_test_m_05(), y_pred_m05, average='weighted')

        prec_r03 = precision_score(Preprocessing_read_csv().y_test_r_03(), y_pred_r03, average='weighted')
        prec_r04 = precision_score(Preprocessing_read_csv().y_test_r_04(), y_pred_r04, average='weighted')
        prec_r05 = precision_score(Preprocessing_read_csv().y_test_r_05(), y_pred_r05, average='weighted')
        prec_m03 = precision_score(Preprocessing_read_csv().y_test_m_03(), y_pred_m03, average='weighted')
        prec_m04 = precision_score(Preprocessing_read_csv().y_test_m_04(), y_pred_m04, average='weighted')
        prec_m05 = precision_score(Preprocessing_read_csv().y_test_m_05(), y_pred_m05, average='weighted')

        train_acc_r03 = accuracy_score(Preprocessing_read_csv().y_train_r_03(), y_pred_train_r03)
        train_acc_r04 = accuracy_score(Preprocessing_read_csv().y_train_r_04(), y_pred_train_r04)
        train_acc_r05 = accuracy_score(Preprocessing_read_csv().y_train_r_05(), y_pred_train_r05)
        train_acc_m03 = accuracy_score(Preprocessing_read_csv().y_train_m_03(), y_pred_train_m03)
        train_acc_m04 = accuracy_score(Preprocessing_read_csv().y_train_m_04(), y_pred_train_m04)
        train_acc_m05 = accuracy_score(Preprocessing_read_csv().y_train_m_05(), y_pred_train_m05)

        overfitting_score_r03 = train_acc_r03 - acc_r03
        overfitting_score_r04 = train_acc_r04 - acc_r04
        overfitting_score_r05 = train_acc_r05 - acc_r05
        overfitting_score_m03 = train_acc_m03 - acc_m03
        overfitting_score_m04 = train_acc_m04 - acc_m04
        overfitting_score_m05 = train_acc_m05 - acc_m05

        acc_score_cross_val_r03 = cross_val_score(model_r03, Preprocessing_read_csv().x_train_r_03(), Preprocessing_read_csv().y_train_r_03(), cv=10, scoring='accuracy')
        prec_score_cross_val_r03 = cross_val_score(model_r03, Preprocessing_read_csv().x_train_r_03(), Preprocessing_read_csv().y_train_r_03(), cv=10, scoring='precision_weighted').mean()
        rec_score_cross_val_r03 = cross_val_score(model_r03, Preprocessing_read_csv().x_train_r_03(), Preprocessing_read_csv().y_train_r_03(), cv=10, scoring='recall_weighted').mean()

        acc_score_cross_val_r04 = cross_val_score(model_r04, Preprocessing_read_csv().x_train_r_04(), Preprocessing_read_csv().y_train_r_04(), cv=10, scoring='accuracy')
        prec_score_cross_val_r04 = cross_val_score(model_r04, Preprocessing_read_csv().x_train_r_04(), Preprocessing_read_csv().y_train_r_04(), cv=10, scoring='precision_weighted').mean()
        rec_score_cross_val_r04 = cross_val_score(model_r04, Preprocessing_read_csv().x_train_r_04(), Preprocessing_read_csv().y_train_r_04(), cv=10, scoring='recall_weighted').mean()

        acc_score_cross_val_r05 = cross_val_score(model_r05, Preprocessing_read_csv().x_train_r_05(), Preprocessing_read_csv().y_train_r_05(), cv=10, scoring='accuracy')
        prec_score_cross_val_r05 = cross_val_score(model_r05, Preprocessing_read_csv().x_train_r_05(), Preprocessing_read_csv().y_train_r_05(), cv=10, scoring='precision_weighted').mean()
        rec_score_cross_val_r05 = cross_val_score(model_r05, Preprocessing_read_csv().x_train_r_05(), Preprocessing_read_csv().y_train_r_05(), cv=10, scoring='recall_weighted').mean()

        acc_score_cross_val_m03 = cross_val_score(model_m03, Preprocessing_read_csv().x_train_m_03(), Preprocessing_read_csv().y_train_m_03(), cv=10, scoring='accuracy')
        prec_score_cross_val_m03 = cross_val_score(model_m03, Preprocessing_read_csv().x_train_m_03(), Preprocessing_read_csv().y_train_m_03(), cv=10, scoring='precision_weighted').mean()
        rec_score_cross_val_m03 = cross_val_score(model_m03, Preprocessing_read_csv().x_train_m_03(), Preprocessing_read_csv().y_train_m_03(), cv=10, scoring='recall_weighted').mean()

        acc_score_cross_val_m04 = cross_val_score(model_m04, Preprocessing_read_csv().x_train_m_04(), Preprocessing_read_csv().y_train_m_04(), cv=10, scoring='accuracy')
        prec_score_cross_val_m04 = cross_val_score(model_m04, Preprocessing_read_csv().x_train_m_04(), Preprocessing_read_csv().y_train_m_04(), cv=10, scoring='precision_weighted').mean()
        rec_score_cross_val_m04 = cross_val_score(model_m04, Preprocessing_read_csv().x_train_m_04(), Preprocessing_read_csv().y_train_m_04(), cv=10, scoring='recall_weighted').mean()

        acc_score_cross_val_m05 = cross_val_score(model_m05, Preprocessing_read_csv().x_train_m_05(), Preprocessing_read_csv().y_train_m_05(), cv=10, scoring='accuracy')
        prec_score_cross_val_m05 = cross_val_score(model_m05, Preprocessing_read_csv().x_train_m_05(), Preprocessing_read_csv().y_train_m_05(), cv=10, scoring='precision_weighted').mean()
        rec_score_cross_val_m05 = cross_val_score(model_m05, Preprocessing_read_csv().x_train_m_05(), Preprocessing_read_csv().y_train_m_05(), cv=10, scoring='recall_weighted').mean()
        
        # Compute confusion matrices
        cm_r03 = sk_confusion_matrix(Preprocessing_read_csv().y_test_r_03(), y_pred_r03)
        cm_r04 = sk_confusion_matrix(Preprocessing_read_csv().y_test_r_04(), y_pred_r04)
        cm_r05 = sk_confusion_matrix(Preprocessing_read_csv().y_test_r_05(), y_pred_r05)
        cm_m03 = sk_confusion_matrix(Preprocessing_read_csv().y_test_m_03(), y_pred_m03)
        cm_m04 = sk_confusion_matrix(Preprocessing_read_csv().y_test_m_04(), y_pred_m04)
        cm_m05 = sk_confusion_matrix(Preprocessing_read_csv().y_test_m_05(), y_pred_m05)
        
        def plot_confusion_matrix(cm, title):
            plt.figure(figsize=(10,7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(title)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            string = base64.b64encode(buf.read()).decode('utf-8')
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)
            buf.close()
            plt.close()
            return uri
        
        cm_r03_image = plot_confusion_matrix(cm_r03, 'Confusion Matrix for R03')
        cm_r04_image = plot_confusion_matrix(cm_r04, 'Confusion Matrix for R04')
        cm_r05_image = plot_confusion_matrix(cm_r05, 'Confusion Matrix for R05')
        cm_m03_image = plot_confusion_matrix(cm_m03, 'Confusion Matrix for M03')
        cm_m04_image = plot_confusion_matrix(cm_m04, 'Confusion Matrix for M04')
        cm_m05_image = plot_confusion_matrix(cm_m05, 'Confusion Matrix for M05')


        title = "Testing Page"
        return render(request, 'testing.html', {
            'title' : title,
            'acc_r03': acc_r03,
            'acc_r04': acc_r04,
            'acc_r05': acc_r05,

            'acc_m03': acc_m03,
            'acc_m04': acc_m04,
            'acc_m05': acc_m05,

            'rec_r03': rec_r03,
            'rec_r04': rec_r04,
            'rec_r05': rec_r05,

            'rec_m03': rec_m03,
            'rec_m04': rec_m04,
            'rec_m05': rec_m05,

            'prec_r03': prec_r03,
            'prec_r04': prec_r04,
            'prec_r05': prec_r05,

            'prec_m03': prec_m03,
            'prec_m04': prec_m04,
            'prec_m05': prec_m05,

            'overfitting_score_r03': overfitting_score_r03,
            'overfitting_score_r04': overfitting_score_r04,
            'overfitting_score_r05': overfitting_score_r05,

            'overfitting_score_m03': overfitting_score_m03,
            'overfitting_score_m04': overfitting_score_m04,
            'overfitting_score_m05': overfitting_score_m05,

            'acc_score_cross_val_r03': acc_score_cross_val_r03,
            'acc_score_cross_val_r04': acc_score_cross_val_r04,
            'acc_score_cross_val_r05': acc_score_cross_val_r05,

            'acc_score_cross_val_m03': acc_score_cross_val_m03,
            'acc_score_cross_val_m04': acc_score_cross_val_m04,
            'acc_score_cross_val_m05': acc_score_cross_val_m05,

            'prec_score_cross_val_r03': prec_score_cross_val_r03,
            'prec_score_cross_val_r04': prec_score_cross_val_r04,
            'prec_score_cross_val_r05': prec_score_cross_val_r05,

            'prec_score_cross_val_m03': prec_score_cross_val_m03,
            'prec_score_cross_val_m04': prec_score_cross_val_m04,
            'prec_score_cross_val_m05': prec_score_cross_val_m05,

            'rec_score_cross_val_r03': rec_score_cross_val_r03,
            'rec_score_cross_val_r04': rec_score_cross_val_r04,
            'rec_score_cross_val_r05': rec_score_cross_val_r05,

            'rec_score_cross_val_m03': rec_score_cross_val_m03,
            'rec_score_cross_val_m04': rec_score_cross_val_m04,
            'rec_score_cross_val_m05': rec_score_cross_val_m05,

            'cm_r03_image': cm_r03_image,
            'cm_r04_image': cm_r04_image,
            'cm_r05_image': cm_r05_image,
            'cm_m03_image': cm_m03_image,
            'cm_m04_image': cm_m04_image,
            'cm_m05_image': cm_m05_image,
        })
    
@login_required(login_url='login')
def dataset(request):
    preprocessing = Preprocessing_read_csv()
    df = preprocessing.read_data()

    title = "Dataset"
    train_data = df.to_dict(orient='records')
    train_columns = df.columns
    train_total_rows = df.shape[0]

    return render(request, 'data.html', {
        'train_data': train_data,
        'train_columns': train_columns,
        'train_total_rows': train_total_rows,
        'title': title
        })

@login_required(login_url='login')
def read_smote_data(request):
    read_csv = Preprocessing_read_csv()
    title = "Dataset SMOTE"
    df_train = read_csv.data_train_model_three()
    df_test = read_csv.data_test_model_three()

    train_data = df_train.to_dict(orient='records')
    train_columns = df_train.columns
    train_total_rows = df_train.shape[0]

    test_data = df_test.to_dict(orient='records')
    test_columns = df_test.columns
    test_total_rows = df_test.shape[0]

    total_data = train_total_rows + test_total_rows
    return render(request, 'data-smote.html', {
        'train_data': train_data,
        'train_columns': train_columns,
        'test_data': test_data,
        'test_columns': test_columns,
        'train_total_rows': train_total_rows,
        'test_total_rows': test_total_rows,
        'total_data': total_data,
        'title': title
    })

@login_required(login_url='login')
def read_real_data(request):
    read_csv = Preprocessing_read_csv()
    title = "Dataset Labelling"
    df_train = read_csv.data_train_real_three()
    df_test = read_csv.data_test_real_three()

    train_data = df_train.to_dict(orient='records')
    train_columns = df_train.columns
    train_total_rows = df_train.shape[0]

    test_data = df_test.to_dict(orient='records')
    test_columns = df_test.columns
    test_total_rows = df_test.shape[0]

    total_data = train_total_rows + test_total_rows
    return render(request, 'data-label.html', {
        'train_data': train_data,
        'train_columns': train_columns,
        'test_data': test_data,
        'test_columns': test_columns,
        'train_total_rows': train_total_rows,
        'test_total_rows': test_total_rows,
        'total_data': total_data,
        'title': title
    })

@login_required(login_url='login')
def classification(request):
    title = "Klasifikasi"
    return render(request, 'klasifikasi.html', {
        'title': title
    })

@login_required(login_url='login')
def prediction(request):
        # Mapping dictionaries
    varietas_mapping = {
        "0.0": "Beras Hitam",
        "0.6": "Ciheran",
        "0.4": "IR 64",
        "0.8": "Mi Kongga",
        "0.2": "Beras Merah",
        "1.0": "Pandan Wangi"
    }

    warna_mapping = {
        "0.67": "Merah",
        "0.0": "Coklat",
        "0.33": "Hitam",
        "1.0": "putih",
    }

    rasa_mapping = {
        "0.0": "Pulen",
        "1.0": "Sangat Pulen"
    }

    musim_mapping = {
        "0.0": "Hujan",
        "1.0": "Kemarau"
    }

    penyakit_mapping = {
        "0.0": "Burung",
        "0.25": "Penggerek Batang",
        "0.5": "Tikus",
        "1.0": "Wereng Hijau",
        "0.75": "Wereng Coklat"
    }

    ph_mapping = {
        "0.0": "2",
        "0.33": "3",
        "0.667": "4",
        "1.0": "5"
    }

    teknik_mapping = {
        "0.0": "Jajar Legowo",
        "1.0": "SRI"
    }

    prediction_mapping = {
        0: "Kelas A",
        1: "Kelas B",
        2: "Kelas C",
        3: "Kelas D"
    }
    if 'prediction' in request.POST:
        # Extract the values from the form
        varietas = request.POST['Varietas']
        warna = request.POST['Warna']
        rasa = request.POST['rasa']
        musim = request.POST['Musim']
        penyakit = request.POST['Penyakit']
        teknik = request.POST['teknik']
        ph = request.POST['PH']
        boron = float(request.POST['boron'])
        fosfor = float(request.POST['fosfor'])

        boron_min = 1.01
        boron_max = 6.05
        fosfor_min = 6.0
        fosfor_max = 7.08

        boron_scaled = (boron - boron_min) / (boron_max - boron_min)
        fosfor_scaled = (fosfor - fosfor_min) / (fosfor_max - fosfor_min)

        # Combine the values into a single array
        # Varietas,fosfor,boron,Warna,rasa,teknik,Musim,Penyakit,PH
        data = np.array([[float(varietas), fosfor_scaled, boron_scaled, float(warna), float(rasa), float(teknik), float(musim), float(penyakit), float(ph)]])
        model = pickle.load(open('model_m03.pkl', 'rb'))

        prediction = model.predict(data)

        # Map the prediction to class name
        prediction_class = prediction_mapping.get(prediction[0], "Unknown")
        # prediction_class = prediction
        # Save the result to the database
        result = ClassificationResult(
            varietas=varietas_mapping[varietas],
            warna=warna_mapping[warna],
            rasa=rasa_mapping[rasa],
            musim=musim_mapping[musim],
            penyakit=penyakit_mapping[penyakit],
            teknik=teknik_mapping[teknik],
            ph=ph_mapping[ph],
            boron=boron,
            fosfor=fosfor,
            prediction=prediction_class
        )
        result.save()
        # Prepare the context for rendering
        context = {
            'varietas': varietas_mapping[varietas],
            'warna': warna_mapping[warna],
            'rasa': rasa_mapping[rasa],
            'musim': musim_mapping[musim],
            'penyakit': penyakit_mapping[penyakit],
            'teknik': teknik_mapping[teknik],
            'ph': ph,
            'boron': boron,
            'fosfor': fosfor,
            'prediction': prediction_class  # Assuming prediction is a single value
        }
    # Fetch all classification results from the database
    results = ClassificationResult.objects.all()
    context['results'] = results
    return render(request, 'klasifikasi.html', context)

def prediction_real_for_guest(request):
    varietas_mapping = {
        "0": "Beras Hitam",
        "3": "Ciheran",
        "2": "IR 64",
        "4": "Mi Kongga",
        "1": "Beras Merah",
        "5": "Pandan Wangi"
    }

    warna_mapping = {
        "2": "Merah",
        "0": "Coklat",
        "1": "Hitam",
        "3": "putih",
    }

    rasa_mapping = {
        "0": "Pulen",
        "1": "Sangat Pulen"
    }

    musim_mapping = {
        "0": "Hujan",
        "1": "Kemarau"
    }

    penyakit_mapping = {
        "0": "Burung",
        "1": "Penggerek Batang",
        "2": "Tikus",
        "3": "Wereng Hijau",
        "4": "Wereng Coklat"
    }

    ph_mapping = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5"
    }

    teknik_mapping = {
        "0": "Jajar Legowo",
        "1": "SRI"
    }

    prediction_mapping = {
        0: "Kelas A",
        1: "Kelas B",
        2: "Kelas C",
        3: "Kelas D"
    }

    if 'predict' in request.POST:
        varietas = request.POST['Varietas']
        warna = request.POST['Warna']
        rasa = request.POST['rasa']
        musim = request.POST['Musim']
        penyakit = request.POST['Penyakit']
        teknik = request.POST['teknik']
        ph = request.POST['PH']
        boron = request.POST['boron']
        fosfor = request.POST['fosfor']

        # Combine the values into a single array
        # Varietas,fosfor,boron,Warna,rasa,teknik,Musim,Penyakit,PH
        data = np.array([[float(varietas), float(fosfor), float(boron), float(warna), float(rasa), float(teknik), float(musim), float(penyakit), float(ph)]])
        model = pickle.load(open('model_r03.pkl', 'rb'))

        pred = model.predict(data)

        # Map the prediction to class name
        prediction = prediction_mapping.get(pred[0], "Unknown")

        return render(request, 'index.html', {
            'prediction': prediction
        })

# Create your views here.
def index(request):
    return render(request, 'index.html')

@login_required(login_url='login')
def adminIndex(request):
    title = "Dashboard"
    return render(request, 'admin-index.html', {
        'title': title
    })

@login_required(login_url='login')
def result(request):
    title = "Result Page"
    return render(request, 'result.html', {
        'title' : title
    })

@login_required(login_url='login')
def excel(request):
    return render(request, 'excel.html')

def excelCoba(request):
    return render(request, 'excel-coba.html')

@login_required(login_url='login')
def excelPrediction(request):
    varietas_mapping = {
        "Beras Hitam": 0.0, 
        "Ciheran": 0.6,
        "IR 64": 0.4,
        "Mi Kongga": 0.8,
        "Beras Merah": 0.2,
        "Pandan Wangi": 1.0
    }

    warna_mapping = {
        "Merah": 0.67,
        "Coklat": 0.0,
        "Hitam": 0.33,
        "putih": 1.0
    }

    rasa_mapping = {
        "Pulen": 0.0,
        "Sangat Pulen": 1.0
    }

    musim_mapping = {
        "Hujan": 0.0,
        "Kemarau": 1.0
    }

    penyakit_mapping = {
        "Burung": 0.0,
        "Penggerek Batang": 0.25,
        "Tikus": 0.5,
        "Wereng Hijau": 1.0,
        "Wereng Coklat": 0.75
    }

    ph_mapping = {
        "2": 0.0,
        "3": 0.33,
        "4": 0.667,
        "5": 1.0
    }

    teknik_mapping = {
        "Jajar Legowo": 0.0,
        "SRI": 1.0
    }

    prediction_mapping = {
        0: "Kelas A",
        1: "Kelas B",
        2: "Kelas C",
        3: "Kelas D"
    }

    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)

        # Read the CSV file
        df = pd.read_csv(fs.path(filename))
        
        # Keep original values for display
        original_df = df.copy()
        
        # Perform the mapping
        df['Varietas'] = df['Varietas'].map(varietas_mapping)
        df['Warna'] = df['Warna'].map(warna_mapping)
        df['rasa'] = df['rasa'].map(rasa_mapping)
        df['Musim'] = df['Musim'].map(musim_mapping)
        df['Penyakit'] = df['Penyakit'].map(penyakit_mapping)
        df['PH'] = df['PH'].map(ph_mapping)
        df['teknik'] = df['teknik'].map(teknik_mapping)

        # Handle possible NaN values by filling them with a default value or strategy
        df.fillna(0, inplace=True)

        # Normalize boron and fosfor using Min-Max Scaler
        boron_min = 1.03
        boron_max = 7.03
        fosfor_min = 1
        fosfor_max = 7

        df['boron'] = (df['boron'] - boron_min) / (boron_max - boron_min)
        df['fosfor'] = (df['fosfor'] - fosfor_min) / (fosfor_max - fosfor_min)

        # Ensure data is in float format
        df = df.astype(float)

        # Prepare the data for prediction (assuming the columns are in the correct order)
        # Varietas,fosfor,boron,Warna,rasa,teknik,Musim,Penyakit,PH
        data = df[['Varietas', 'fosfor', 'boron', 'Warna', 'rasa', 'teknik', 'Musim', 'Penyakit', 'PH']].values
        
        # Load the model
        model = pickle.load(open('model_m03.pkl', 'rb'))

        # Predict
        predictions = model.predict(data)

        # Map predictions to class names
        original_df['Prediction'] = [prediction_mapping[pred] for pred in predictions]

        # Convert DataFrame to list of columns and rows for template
        columns = original_df.columns.tolist()
        rows = original_df.values.tolist()

        # Render the results in the same template
        context = {
            'columns': columns,
            'rows': rows
        }

        return render(request, 'excel.html', context)

    return render(request, 'excel.html')

@login_required(login_url='login')
def download_csv(request):
    file_path = os.path.join(settings.BASE_DIR, 'model/data/format-import-excel.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename={}'.format(smart_str(os.path.basename(file_path)))
            response['Content-Length'] = os.path.getsize(file_path)
            return response
    else:
        # Jika file tidak ditemukan, bisa kembalikan respons 404
        return HttpResponse(status=404)