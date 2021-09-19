from django.db import models

# Create your models here.
class Image(models.Model):
    photo = models.ImageField(upload_to = "myimage")
    date = models.DateTimeField(auto_now_add=True)

class myuploadfile(models.Model):
    specialId = models.IntegerField()
    name = models.CharField(null= True ,max_length=200)
    myfiles = models.FileField(upload_to="myimage")
    dcp = models.FileField(upload_to="myimage")
    gf = models.FileField(upload_to="myimage" )
    cyclicgan = models.FileField(upload_to="myimage")
    NovelIdea = models.FileField(upload_to="myimage")
    dcp_psnr = models.FloatField(max_length=200, null= True)
    gf_psnr = models.FloatField(max_length=200 , null= True)
    cyclicgan_psnr = models.FloatField(default = 0, max_length=200 , null= True)
    NovelIdea_psnr = models.FloatField(default = 0, max_length=200 , null= True)
    dcp_ssim = models.FloatField(max_length=200 , null= True)
    gf_ssim = models.FloatField(max_length=200 , null= True)
    cyclicgan_ssim = models.FloatField(default = 0, max_length=200 , null= True)
    NovelIdea_ssim = models.FloatField(default = 0, max_length=200 , null= True)


class PSNR(models.Model):
    imageID = models.CharField(primary_key = True , max_length=200)
    dcp = models.FloatField(max_length=200)
    gf = models.FloatField(max_length=200)
    cyclicgan = models.FloatField(default = 0, max_length=200 , null= True)
    NovelIdea = models.FloatField(default = 0, max_length=200 , null= True)


class SSIM(models.Model):
    imageID = models.CharField(primary_key = True , max_length=200)
    dcp = models.FloatField(max_length=200)
    gf = models.FloatField(max_length=200)
    cyclicgan = models.FloatField(default = 0, max_length=200 , null= True)
    NovelIdea = models.FloatField(default = 0, max_length=200 , null= True)

class PARAMS(models.Model):
    imageID = models.CharField(primary_key = True , max_length=200)
    pdcp = models.FloatField(max_length=200)
    pgf = models.FloatField(max_length=200)
    pcyclicgan = models.FloatField(default = 0, max_length=200 , null= True)
    pNovelIdea = models.FloatField(default = 0, max_length=200 , null= True)
    sdcp = models.FloatField(max_length=200)
    sgf = models.FloatField(max_length=200)
    scyclicgan = models.FloatField(default = 0, max_length=200 , null= True)
    sNovelIdea = models.FloatField(default = 0, max_length=200 , null= True)

class Feedback(models.Model):
    Name = models.CharField(max_length=200)
    Email = models.CharField(max_length=200, null = True)
    Message = models.CharField(max_length=200) 

class Contact(models.Model):
    Name = models.CharField(max_length=200)
    Email = models.CharField(max_length=200, null = True)
    Message = models.CharField(max_length=200) 