# Generated by Django 3.2.23 on 2024-06-13 08:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0002_auto_20240613_1531'),
    ]

    operations = [
        migrations.AlterField(
            model_name='classificationresult',
            name='musim',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='classificationresult',
            name='penyakit',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='classificationresult',
            name='prediction',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='classificationresult',
            name='rasa',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='classificationresult',
            name='teknik',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='classificationresult',
            name='varietas',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='classificationresult',
            name='warna',
            field=models.CharField(max_length=100),
        ),
    ]
