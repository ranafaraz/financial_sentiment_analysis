# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # ... other url patterns ...
    path('upload/', views.upload_file, name='upload_file'),
    # path('success/', views.success, name='success_url'),  # If you created a success view
]
