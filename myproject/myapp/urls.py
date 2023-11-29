# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # ... other url patterns ...
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload_file'),
    path('view-results/', views.view_results, name='view_results'),

    # path('success/', views.success, name='success_url'),  # If you created a success view
]
