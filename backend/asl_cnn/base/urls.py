from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home, name='index'),
    path('predict/', views.predict, name='predict'),
]