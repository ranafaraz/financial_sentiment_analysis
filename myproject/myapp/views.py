# myapp/views.py

from django.shortcuts import render, redirect
from .forms import FileUploadForm
from .models import UploadedFile
from .models import AnalysisResult

def home(request):
    return render(request, 'myapp/home.html')

def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success_url')  # Redirect to a success page
    else:
        form = FileUploadForm()
    return render(request, 'myapp/upload.html', {'form': form})

def view_results(request):
    results = AnalysisResult.objects.all()  # Retrieve all results from the database
    return render(request, 'myapp/view_results.html', {'results': results})
