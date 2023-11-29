# myapp/forms.py

from django import forms
from .models import UploadedFile  # Import the UploadedFile model

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['datafile']
