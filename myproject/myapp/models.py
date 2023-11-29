from django.db import models

# Create your models here.
class UploadedFile(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    datafile = models.FileField(upload_to='uploads/')

class AnalysisResult(models.Model):
    blended_classifiers = models.TextField()
    accuracy = models.FloatField()
    kappa = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    confusion_matrix = models.TextField()
    execution_time = models.FloatField()
    total_classifiers = models.IntegerField()
    total_features = models.IntegerField()
    training_data_size = models.IntegerField()
    test_data_size = models.IntegerField()
    random_state = models.IntegerField()
    preprocessing = models.BooleanField()
    smote = models.BooleanField()
    total_cpu_cores = models.IntegerField()
    cpu_usage = models.FloatField()
    total_ram = models.FloatField()
    memory_usage = models.TextField()
    processor_type = models.TextField()
    os_name = models.TextField()




# python manage.py makemigrations
# python manage.py migrate
