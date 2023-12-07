# myapp/views.py

from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import render, redirect
from .forms import *
from .models import *
from .analysis import *

def home(request):
    return render(request, 'myapp/home.html')

# Upload Dataset
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('dataset_list')  # Redirect to a page that lists datasets
    else:
        form = DatasetUploadForm()
    return render(request, 'myapp/upload_dataset.html', {'form': form})

# Dataset Listing
def dataset_list(request):
    datasets = Dataset.objects.all()
    return render(request, 'myapp/dataset_list.html', {'datasets': datasets})

## Perform Analysis Web Page
def perform_analysis(request):
    if request.method == 'POST':
        form = ClassifierDatasetForm(request.POST)
        if form.is_valid():
            # Process the form data
            # Redirect to the analysis results page after processing
            return redirect('view_results')
    else:
        form = ClassifierDatasetForm()

    return render(request, 'myapp/perform_analysis.html', {'form': form})
####################################################################################
def view_results(request):
    return render(request, 'myapp/view_results.html')

def get_results(request):
    draw = int(request.GET.get('draw', default=1))  # Used by DataTables
    start = int(request.GET.get('start', default=0))
    length = int(request.GET.get('length', default=10))
    search_value = request.GET.get('search[value]', default='')

    # Query based on DataTables parameters
    results = AnalysisResult.objects.all()
    if search_value:
        results = results.filter(blended_classifiers__icontains=search_value)  # Example of filtering

    # Total number of records before filtering
    total = results.count()

    # Paginator
    paginator = Paginator(results, length)
    page_number = start // length + 1
    page = paginator.page(page_number)

    # Preparing response
    data = list(page.object_list.values(
        'blended_classifiers', 'accuracy', 'kappa', 'precision', 'recall', 'f1_score',
        'confusion_matrix', 'execution_time', 'total_classifiers', 'total_features',
        'training_data_size', 'test_data_size', 'random_state', 'preprocessing', 'smote',
        'total_cpu_cores', 'cpu_usage', 'total_ram', 'memory_usage', 'processor_type', 'os_name'

    ))

    response = {
        'draw': draw,
        'recordsTotal': total,
        'recordsFiltered': total,
        'data': data
    }

    return JsonResponse(response)
