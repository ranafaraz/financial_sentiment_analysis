# myapp/views.py

from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import render, redirect
from .forms import *
from .models import *
from .analysis import perform_analysis
from django.conf import settings
import os

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
def perform_analysis_view(request):
    datasets = Dataset.objects.all()
    if request.method == 'POST':
        selected_dataset_url = request.POST.get('dataset')
        # Assume the selected_dataset_url is something like "/media/datasets/myfile.csv"
        relative_file_path = selected_dataset_url.replace(settings.MEDIA_URL, '', 1)

        # Construct the full file system path by appending the relative path to MEDIA_ROOT
        full_file_path = os.path.join(settings.MEDIA_ROOT, relative_file_path)

        # Get the list of selected classifiers
        selected_classifiers = request.POST.getlist('classifiers')

        # Map the selected classifier names to their corresponding objects
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SDG': SGDClassifier(),
            'Random Forest': RandomForestClassifier(max_depth=10),
            'Support Vector Machine': SVC(),
            'Naive Bayes': MultinomialNB(),
            'Decision Tree': DecisionTreeClassifier(),
            'Bagging': BaggingClassifier(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Extreme Gradient Boosting (XGBoost)': XGBClassifier()
        }
        selected_classifier_objects = [classifiers_dict[name] for name in selected_classifiers if name in classifiers_dict]

        # Debugging: Print the classifier objects
        print("Classifier Objects:", selected_classifier_objects)

        # Call the perform_analysis method with the file path and selected classifiers
        # Debugging: Print the file path
        print("File Path:", file_path)

        # Call the perform_analysis method with the file path and selected classifiers
        # perform_analysis(full_file_path, selected_classifier_objects)

        # Redirect to results or another page as needed
        return redirect('view_results')
    else:
        datasets = Dataset.objects.all()
        # Include other necessary forms
        return render(request, 'myapp/perform_analysis.html', {'datasets': datasets})
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
