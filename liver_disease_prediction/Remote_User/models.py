from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class disease_prediction(models.Model):

    Pid= models.CharField(max_length=30000)
    Age= models.CharField(max_length=300)
    Gender= models.CharField(max_length=300)
    Total_Bilirubin= models.CharField(max_length=300)
    Direct_Bilirubin= models.CharField(max_length=300)
    Alkaline_Phosphotase= models.CharField(max_length=300)
    Alamine_Aminotransferase= models.CharField(max_length=300)
    Aspartate_Aminotransferase= models.CharField(max_length=300)
    Total_Protiens= models.CharField(max_length=300)
    Albumin= models.CharField(max_length=300)
    Albumin_and_Globulin_Ratio= models.CharField(max_length=300)
    prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



