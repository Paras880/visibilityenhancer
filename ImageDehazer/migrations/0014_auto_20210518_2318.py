# Generated by Django 3.2 on 2021-05-18 17:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ImageDehazer', '0013_params'),
    ]

    operations = [
        migrations.AlterField(
            model_name='myuploadfile',
            name='NovelIdea',
            field=models.FileField(upload_to='myimage'),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='cyclicgan',
            field=models.FileField(upload_to='myimage'),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='dcp',
            field=models.FileField(upload_to='myimage'),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='gf',
            field=models.FileField(upload_to='myimage'),
        ),
    ]
