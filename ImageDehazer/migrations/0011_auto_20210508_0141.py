# Generated by Django 3.2 on 2021-05-07 20:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ImageDehazer', '0010_remove_myuploadfile_f_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='myuploadfile',
            name='NovelIdea',
            field=models.FileField(null=True, upload_to='myimage'),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='cyclicgan',
            field=models.FileField(null=True, upload_to='myimage'),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='dcp',
            field=models.FileField(null=True, upload_to='myimage'),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='gf',
            field=models.FileField(null=True, upload_to='myimage'),
        ),
    ]
