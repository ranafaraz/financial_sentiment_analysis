from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=5000)
    file = models.FileField(upload_to='datasets/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class AnalysisResult(models.Model):
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)
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
    created_at = models.DateTimeField(auto_now_add=True)




# python manage.py makemigrations
# python manage.py migrate
