# Generated by Django 5.1.4 on 2024-12-20 02:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BeachPrediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('beach_name', models.CharField(max_length=100)),
                ('timestamp', models.DateTimeField()),
                ('sea_surface_temp', models.FloatField()),
                ('air_temp', models.FloatField()),
                ('wind_speed', models.FloatField()),
                ('wave_height', models.FloatField()),
                ('uv_index', models.FloatField()),
                ('activity_level', models.CharField(max_length=50)),
            ],
        ),
    ]
