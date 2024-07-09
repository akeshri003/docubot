from django.db import models

# Create your models here.
class Site(models.Model):
    company_name = models.CharField(max_length=100)
    url = models.URLField()
    output = models.TextField(blank = True, null = True)

    def __str__(self):
        return self.company_name + ' '