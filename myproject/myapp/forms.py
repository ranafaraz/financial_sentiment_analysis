# myapp/forms.py

from django import forms
from .models import *


class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']  # Ensure this matches the field name in the model

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file', 'description']

class ClassifierDatasetForm(forms.Form):
    classifiers = forms.MultipleChoiceField(
        choices=[('classifier1', 'Classifier 1'), ('classifier2', 'Classifier 2')],
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
    dataset = forms.ModelChoiceField(
        queryset=UploadedFile.objects.all(),
        required=True
    )
