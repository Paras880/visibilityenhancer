from django.contrib import admin
from .models import Feedback, Image, PARAMS, PSNR, SSIM, myuploadfile, Feedback, Contact 
from import_export.admin import ImportExportModelAdmin

# Register your models here.
@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id' , 'photo' , 'date']

@admin.register(PSNR)
class PSNRAdmin(admin.ModelAdmin):
    list_display = ['imageID' , 'dcp' , 'gf' , 'cyclicgan' , 'NovelIdea']

@admin.register(SSIM)
class SSIMAdmin(admin.ModelAdmin):
    list_display = ['imageID' , 'dcp' , 'gf' , 'cyclicgan' , 'NovelIdea']

@admin.register(PARAMS)
class PARAMSAdmin(admin.ModelAdmin):
    list_display = ['imageID' , 'pdcp' , 'pgf' , 'pcyclicgan' , 'pNovelIdea', 'sdcp' , 'sgf' , 'scyclicgan' , 'sNovelIdea']

@admin.register(myuploadfile)
class myuploadfileAdmin(admin.ModelAdmin):
    list_display = ['id' ,'name' , 'specialId','myfiles' , 'dcp', 'gf' , 'cyclicgan' , 'NovelIdea', 'dcp_psnr' , 'gf_psnr' , 'cyclicgan_psnr' , 'NovelIdea_psnr' , 'dcp_ssim' , 'gf_ssim' , 'cyclicgan_ssim' , 'NovelIdea_ssim']

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ['id' , 'Name' , 'Email' , 'Message']

@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ['id' , 'Name' , 'Email' , 'Message']