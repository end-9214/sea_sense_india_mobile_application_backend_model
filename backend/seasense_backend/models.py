from django.db import models

class BeachPrediction(models.Model):
    beach_name = models.CharField(max_length=100)
    timestamp = models.DateTimeField()
    sea_surface_temp = models.FloatField()
    air_temp = models.FloatField()
    wind_speed = models.FloatField()
    wave_height = models.FloatField()
    uv_index = models.FloatField()
    activity_level = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.beach_name} - {self.timestamp}"