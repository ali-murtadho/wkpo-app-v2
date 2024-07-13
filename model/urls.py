from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('login', views.login_view, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout_view, name='logout'),
    path('guest-predict-page', views.render_predict_guest, name='guest_predict_page'),
    path('prediction-guest', views.prediction_real_for_guest, name='prediction_guest'),
    
    path('home', views.adminIndex, name='home'),
    path('dataset', views.dataset, name='data'),
    path('model', views.result, name='model'),
    path('classification', views.classification, name='classification'),
    path('excel', views.excel, name='excel'),
    path('dataset-after-smote', views.read_smote_data, name='data_smote'),
    path('dataset-before-smote', views.read_real_data, name='data_label'),
    path('training', views.training, name='training'),
    path('testing', views.testing, name='testing'),
    path('prediction', views.prediction, name='prediction'),
    path('download-csv', views.download_csv, name='download-csv'),
    path('excel-classification', views.excelPrediction, name='excel-classification'),
]