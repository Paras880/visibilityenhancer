# Generated by Django 3.2 on 2021-05-12 01:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ImageDehazer', '0011_auto_20210508_0141'),
    ]

    operations = [
        migrations.AddField(
            model_name='myuploadfile',
            name='specialId',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
