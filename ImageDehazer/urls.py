from django.urls import path
from ImageDehazer import views

urlpatterns = [
    path("", views.home, name="home"),
    path("feedback", views.feedback, name="feedaback"),
    path("contact", views.contact, name="contact"),
    path("qa", views.qa, name="qa"),
    path("Dehazingmethods", views.Dehazingmethods, name="Dehazingmethods"),
    #path("",views.index),
    path("get",views.get , name = "get"),
    path("datasetpdf",views.datasetpdf , name = "datasetpdf"),
    # path("download",views.download , name = "download"),
    path("uploads",views.send_files,name="uploads"),
    path("n", views.new, name="new"),
    path("Sotsindoor", views.datasets1, name="datasets1"),
    path("Sotsoutdoor", views.datasets2, name="datasets2"),
    path("hstsreal", views.dataseth1, name="dataseth1"),
    path("hstssynth", views.dataseth2, name="dataseth2"),
    path("enlarge", views.enlarge, name="enlarge"),
    path("dehaze", views.dehaze, name="dehaze"),
    path("upload", views.upload, name="upload"),
    path("simple_upload", views.simple_upload, name="simple_upload"),
]