# Generated by Django 3.2 on 2021-05-07 06:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ImageDehazer', '0007_auto_20210507_1049'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ssim',
            name='NovelIdea',
            field=models.FloatField(default=0, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='ssim',
            name='cyclicgan',
            field=models.FloatField(default=0, max_length=200, null=True),
        ),
    ]
