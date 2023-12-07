# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [

    path('', views.home, name='home'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('view-results/', views.view_results, name='view_results'),
    path('api/get-results/', views.get_results, name='get_results'),
    path('perform_analysis/', views.perform_analysis_view, name='perform_analysis'),



    # path('success/', views.success, name='success_url'),  # If you created a success view
]
