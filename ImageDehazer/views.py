from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageForm
from .models import Image, PARAMS
from .models import PSNR
from .models import SSIM , myuploadfile , Feedback, Contact
from .resources import PSNRResource
from .resources import SSIMResource
from .resources import PARAMSResource
from tablib import Dataset
from .utils import get_plot
from django.db.models import Q
# from IO import StringIO
# import cStringIO as cs
import zipfile as z
from zipfile import ZipFile
import random
from PIL import Image as im
import cv2;
import math;
import numpy as np;
import sys
import os
import datetime
# Create your views here.

from . models import myuploadfile

# pdf file viewer

from django.http import HttpResponse
from django.views.generic import View

from .utils import render_to_pdf #created in step 4


# Create your views here.
# def download(request):     
    
#     f = cs.StringIO() # or a file on disk in write-mode
#     zf = z.ZipFile(f, 'w', z.ZIP_DEFLATED)

#     # in_memory = StringIO()
#     # zip = ZipFile(in_memory, "a")

#     dcpphoto = request.POST.get("dcpphoto")
#     gfphoto = request.POST.get("gfphoto")   
    
#     zf.writestr(dcpphoto , "darkchannelprior")
#     zf.writestr(gfphoto , "guidedfilter")
    
#     # fix for Linux zip files read in Windows
#     for file in zf.filelist:
#         file.create_system = 0    
        
#     zf.close()

#     response = HttpResponse(mimetype="application/zip")
#     response["Content-Disposition"] = "attachment; filename=two_files.zip"
    
#     # in_memory.seek(0)    
#     # response.write(in_memory.read())
    
#     return response
def contact(request):
    if request.method == "POST" :
        name = request.POST.get("name")
        EmailAddress = request.POST.get("email")
        message = request.POST.get("message")
    
    if (name != None and EmailAddress != None and message != None):
        Contact(Name = name , Email = EmailAddress , Message= message).save()
    
    return render(request, "ImageDehazer/index1.html")

def feedback(request):
    if request.method == "POST" :
        name = request.POST.get("name")
        EmailAddress = request.POST.get("email")
        message = request.POST.get("message")
    print(name , EmailAddress , message)


    if (name != None and EmailAddress != None and message != None):
        Feedback(Name = name , Email = EmailAddress , Message= message).save()
    
    return render(request, "ImageDehazer/index1.html")





def index(request):
    context = {
        "data":myuploadfile.objects.all(),
    }
    return render(request,"ImageDehazer/index.html",context)

def Psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def Ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    #print(b , g , r)
    dc = cv2.min(cv2.min(r,g),b);
    #print(dc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

original = ""
name = ""
psnr = 0
ssim = 0
method = ""

def send_files(request):
    if request.method == "POST" :
        name = request.POST.get("filename")
        myfile = request.FILES.getlist("uploadfoles")
        method = request.POST.get("method")
        print("method - ", method)
        if (len(myfile) == 1 and method != "all"):
            Image(photo=myfile[0]).save()
            img = Image.objects.all()
            originalpdf = img.last().photo.path
            original = img.last().photo.url
            src = cv2.imread("." + img.last().photo.url);
            I = src.astype('float64')/255;

            dark = DarkChannel(I,15);
            A = AtmLight(I,dark);
            te = TransmissionEstimate(I,A,15);
            t = TransmissionRefine(src,te);

            if(method == "DCP"):
                J = Recover(I,te,A,0.1);
            elif(method == "GUIDED"):
                J = Recover(I,t,A,0.1);

            
            name = "."+ img.last().photo.url.split(".")[0] + str(random.randint(0,9)) + ".png"
            a = originalpdf.split("\\")
            print(a)
            imagename = name.split("/")[-1]
            a[-1] = name.split("/")[-1]
            dehazedpdf = "\\".join(a)

            print("Exception")
            print(dehazedpdf)
            print(originalpdf)
            print("exceptin")
            cv2.imwrite(name,J*255);

            psnr = round(Psnr(src , J*255) , 3)
            ssim = round(Ssim(src , J*255) , 3)
            print(imagename)
            return render(request , "ImageDehazer/Single-Results.html" , {'imagename':imagename , 'dehazedpdf':dehazedpdf , 'originalpdf': originalpdf ,  'dehazedphoto' : name , 'originalphoto' : original , 'psnr': psnr , 'ssim' :ssim , "method": method})

        else:
            new_id = random.randint(1,1000000)
            no_files = len(myfile)
            for f in myfile:
                myuploadfile(specialId = new_id , name = f,  myfiles=f).save()
                
            
            new_images =  myuploadfile.objects.filter(specialId = new_id)
            
            for i in new_images:
                
                src = cv2.imread("." + i.myfiles.url);
                I = src.astype('float64')/255;

                dark = DarkChannel(I,15);
                A = AtmLight(I,dark);
                te = TransmissionEstimate(I,A,15);
                t = TransmissionRefine(src,te);

                if(method == "DCP"):
                    J = Recover(I,te,A,0.1);
                elif(method == "GUIDED"):
                    J = Recover(I,t,A,0.1);
                elif(method == "all"):
                    #DCP
                    J = Recover(I,te,A,0.1);
                    name = i.myfiles.url.split("/")[-2] + "/" +i.myfiles.url.split("/")[-1] + "DCP" + ".jpeg"
                    save_name = "./media/"+name
                    cv2.imwrite(save_name,J*255)
                    procecssed = myuploadfile.objects.get(myfiles = i.myfiles)
                    procecssed.dcp = name
                    procecssed.dcp_psnr = round(Psnr(src , J*255) , 3)
                    procecssed.dcp_ssim = round(Ssim(src , J*255) , 3)
                    procecssed.save()

                    #GUIDED FILTER
                    J = Recover(I,t,A,0.1);
                    name = i.myfiles.url.split("/")[-2] + "/" +i.myfiles.url.split("/")[-1] + "GF" + ".jpeg"
                    save_name = "./media/"+name
                    cv2.imwrite(save_name,J*255)
                    procecssed = myuploadfile.objects.get(myfiles = i.myfiles)
                    procecssed.gf = name
                    procecssed.gf_psnr = round(Psnr(src , J*255) , 3)
                    procecssed.gf_ssim = round(Ssim(src , J*255) , 3)
                    procecssed.save()
                    print("one")

                    if(len(myfile) == 1):
                        all_images =  myuploadfile.objects.filter(specialId = new_id)
                        print(all_images)
                        return render(request , "ImageDehazer/Multiple-Results.html" , {"files" : all_images[0]} )
                
                if(method != "all"):
                    print("all")
                    name = i.myfiles.url.split("/")[-2] + "/" +i.myfiles.url.split("/")[-1] + method + ".jpeg"
                    save_name = "./media/"+name
                    
                    cv2.imwrite(save_name,J*255)
                    procecssed = myuploadfile.objects.get(myfiles = i.myfiles)
                    print(procecssed)

                    if(method == "DCP"):
                        procecssed.dcp = name
                        procecssed.dcp_psnr = round(Psnr(src , J*255) , 3)
                        procecssed.dcp_ssim = round(Ssim(src , J*255) , 3)
                    elif(method == "GUIDED"):
                        procecssed.gf = name
                        procecssed.gf_psnr = round(Psnr(src , J*255) , 3)
                        procecssed.gf_ssim = round(Ssim(src , J*255) , 3)
                    procecssed.save()

            dcp_images =  myuploadfile.objects.filter(specialId = new_id)
            if(method == "all"):
                Everything = myuploadfile.objects.filter(specialId = new_id)
                return render(request , "ImageDehazer/Multiplemethods.html" , {"files" : Everything} )
            else:
                return render(request , "ImageDehazer/Singlemethod.html" , {"method" : method , "files" : dcp_images} )
                # uploaded_dataset.append(f)
            # print(uploaded_dataset)
        
        # for img in myuploadfile.objects.all():
        #     print(img)
        #     if img.myfiles in uploaded_dataset:
        #         print(img.myfiles)
        
    #         src = cv2.imread("." + myuploadfile.myfiles.name );
    #         I = src.astype('float64')/255;

    #         dark = DarkChannel(I,15);
    #         A = AtmLight(I,dark);
    #         te = TransmissionEstimate(I,A,15);
    #         t = TransmissionRefine(src,te);
    #         J = Recover(I,te,A,0.1);
    #         Jgf = Recover(I,t,A,0.1);
        
    # name = "."+ img.last().photo.url.split(".")[0] + str(random.randint(0,9)) + ".png"
    # cv2.imwrite(name,J*255);
    # cv2.imwrite(name,J*255);

    # psnr = Psnr(src , J*255)
    # ssim = Ssim(src , J*255)
    return redirect("/")

def simple_upload(request):
    if request.method == 'POST':
        ssim_resource = SSIMResource()
        dataset = Dataset()
        new_ssim = request.FILES['myFile']
        print(new_ssim)

        imported_data = dataset.load(new_ssim.read(),format='xlsx')
        #print(imported_data)
        for data in imported_data:
            print(data)
            value = SSIM(
        		 data[0],
        		 data[1],
                 data[2],
        		 data[3],
        		 data[4],
            )
            value.save()       
        
        #result = person_resource.import_data(dataset, dry_run=True)  # Test the data import

        #if not result.has_errors():
        #    person_resource.import_data(dataset, dry_run=False)  # Actually import now

    return render(request, 'ImageDehazer/upload.html')

def upload(request):
    return render(request , "ImageDehazer/upload.html")

# def home(request):
#     form  = ImageForm()
#     # img = Image.objects.all()
#     #print("images" , img[0].photo)
#     return render(request , "ImageDehazer/form.html" ,{'form':form })
def home(request):
    first  = Feedback.objects.get(id = 2)
    
    feed = Feedback.objects.filter(~Q(id=2))

    return render(request, "ImageDehazer/index1.html" , {"first":first , "feed":feed})
    
def Dehazingmethods(request):

    return render(request, "ImageDehazer/Dehazing-Methods.html")
    

def datasets1(request):
    List = os.listdir("C:/Users/PARAS/Desktop/hello_django/ImageDehazer/static/ImageDehazer/SOTS/indoor/hazy")
    images = []
    for i in List:
        images.append("ImageDehazer/SOTS/indoor/hazy/" + i)
    length = len(images)
    image = zip(images , List)
    params = PARAMS.objects.all()
    return render(request,'ImageDehazer/Reside-Dataset.html', {'image' : image , "params" : params})
def datasets2(request):
    List = os.listdir("C:/Users/PARAS/Desktop/hello_django/ImageDehazer/static/ImageDehazer/SOTS/outdoor/hazy")
    images = []
    for i in List:
        images.append("ImageDehazer/SOTS/outdoor/hazy/" + i)
    length = len(images)
    image = zip(images , List)
    params = PARAMS.objects.all()
    return render(request,'ImageDehazer/Reside-Dataset.html', {'image' : image , "params" : params})

def dataseth1(request):
    List = os.listdir("C:/Users/PARAS/Desktop/hello_django/ImageDehazer/static/ImageDehazer/HSTS/real-world")
    images = []
    for i in List:
        images.append("ImageDehazer/HSTS/real-world/" + i)
    length = len(images)
    image = zip(images , List)
    params = PARAMS.objects.all()
    return render(request,'ImageDehazer/Reside-Dataset.html', {'image' : image , "params" : params})

def dataseth2(request):
    List = os.listdir("C:/Users/PARAS/Desktop/hello_django/ImageDehazer/static/ImageDehazer/HSTS/synthetic/synthetic")
    images = []
    for i in List:
        images.append("ImageDehazer/HSTS/synthetic/synthetic/" + i)
    length = len(images)
    image = zip(images , List)
    params = PARAMS.objects.all()
    return render(request,'ImageDehazer/Reside-Dataset.html', {'image' : image , "params" : params})

def enlarge(request):
    imageId =  request.POST.get('imageId')
    print(imageId)
    imageId = imageId[8:]
    splitwords = imageId.split("/")
    image = splitwords[-1]
    params = PARAMS.objects.all()

    valssim1 = 0
    valssim2 = 0
    valpsnr1 = 0
    valpsnr2 = 0
    print(image)

    for i in params:
        print(i)
        if i.imageID == image:
            valpsnr1 = i.pdcp
            valpsnr2 = i.pgf
            valssim1 = i.sdcp
            valssim2 = i.sgf

    base = "C:\\Users\\PARAS\\Desktop\\hello_django\\ImageDehazer\\static\\ImageDehazer"
    # algo for attaching corresponding image results
    imagepdf = ""
    imagedcp = ""
    imagedcppdf = ""
    imagegf = ""
    imagegfpdf = ""

    if (splitwords[-3] == "HSTS" or splitwords[-3] == "synthetic"):
        if (splitwords[-2] == "real-world"):
            imagepdf = base + "\\HSTS\\real-world"+ image
            splitwords[-2] = "real-worldoutputdcp"
            imagedcp ='/'.join(splitwords)
            imagedcppdf = base + "\\HSTS\\real-worldoutputdcp\\"+image
            splitwords[-2] = "real-worldoutputgf"
            imagegf ='/'.join(splitwords)
            imagegfpdf = base + "\\HSTS\\real-worldoutputgf\\"+ image
        elif (splitwords[-2] == "synthetic"):
            imagepdf = base + "\\HSTS\\synthetic\\synthetic\\"+image
            splitwords[-2] = "syntheticoutputdcp"
            imagedcp ='/'.join(splitwords)
            imagedcppdf = base + "\\HSTS\\synthetic\\syntheticoutputdcp\\"+image
            splitwords[-2] = "syntheticoutputgf"
            imagegf ='/'.join(splitwords)
            imagegfpdf = base + "\\HSTS\\synthetic\\syntheticoutputgf\\"+image
    elif (splitwords[-3] == "outdoor" or splitwords[-3] == "indoor"):
        if (splitwords[-3] == "outdoor"):
            imagepdf = base + "\\SOTS\\outdoor\\hazy\\"+image
            if(splitwords[-2] == "hazyoutputdcp"):
                imagedcppdf = base +"\\SOTS\\outdoor\\hazyoutputdcp\\"+image
            elif(splitwords[-2] == "hazyoutputgf"):
                imagegfpdf = base +"\\SOTS\\outdoor\\hazyoutputgf\\"+image
        elif (splitwords[-3] == "indoor"):
            imagepdf = base + "\\SOTS\\indoor\\hazy\\"+image
            if(splitwords[-2] == "hazyoutputdcp"):
                imagedcppdf = base +"\\SOTS\\indoor\\hazyoutputdcp\\"+image
            elif(splitwords[-2] == "hazyoutputgf"):
                imagegfpdf = base +"\\SOTS\\indoor\\hazyoutputgf\\"+image
        splitwords[-2] = "hazyoutputdcp"
        imagedcp ='/'.join(splitwords)
        splitwords[-2] = "hazyoutputgf"
        imagegf ='/'.join(splitwords)


    print(imageId)
    print(imagedcp)
    print(imagegf)

    # qs = PARAMS.objects.filter(imageID = image)
    # x = [qs[0].pdcp , qs[0].pgf , 0 , 0]
    # y = [qs[0].sdcp , qs[0].sgf , 0 , 0] 
    # chart = get_plot(x ,y)
    # print(chart)
    chart = ""
    return render(request,'ImageDehazer/Dataset-Results.html' , { "chart" : chart , "imagegfpdf":imagegfpdf , "imagedcppdf" :imagedcppdf , "imagepdf":imagepdf , "imagename" : image , "imageId" : imageId , "imagedcp" : imagedcp , "imagegf" : imagegf , "psnrdcp" : valpsnr1 , "psnrgf" : valpsnr2 , "ssimdcp": valssim1 , "ssimgf" : valssim2})

def new(request):
    method = request.POST.get('method')
    print(method)
    return render(request,'ImageDehazer/Upload-Image (2).html' ,{'method': method})

def Psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def Ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    #print(b , g , r)
    dc = cv2.min(cv2.min(r,g),b);
    #print(dc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

original = ""
name = ""
psnr = 0
ssim = 0
method = ""

def dehaze(request):

    if request.method == 'POST':
        print(request.POST)
        print(request.FILES)
        form = ImageForm(request.POST , request.FILES)
        print(form)
        if form.is_valid():
            form.save()
    

    dcp = request.POST.get('dcp' , "off")
    print(dcp)
    guidedfilter = request.POST.get('guidedfilter' , "off")
    print(guidedfilter)
    nonlocalprior = request.POST.get('nonlocalprior' , "off")
    print(nonlocalprior)
    # try:
    #     fn = sys.argv[1] 
    # except:
    #     fn = photo
    img = Image.objects.all()
    print("nbn nmn")
    print(img)
    def nothing(*argv):
        pass

    print(img.last().photo.path)
    originalpdf = img.last().photo.path
    original = img.last().photo.url
    src = cv2.imread("." + img.last().photo.url);
    I = src.astype('float64')/255;

    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);

    if(dcp == "on"):
        method = "DCP"
        J = Recover(I,te,A,0.1);
    elif(guidedfilter == "on"):
        method = "GUIDED FILTER"
        J = Recover(I,t,A,0.1);
    
    name = "."+ img.last().photo.url.split(".")[0] + str(random.randint(0,9)) + ".png"
    a = originalpdf.split("\\")
    print(a)
    print(name.split("/")[-1])
    a[-1] = name.split("/")[-1]
    dehazedpdf = "\\".join(a)

    print("Exception")
    print(dehazedpdf)
    print(originalpdf)
    print("exceptin")
    cv2.imwrite(name,J*255);

    psnr = Psnr(src , J*255)
    ssim = Ssim(src , J*255)

    return render(request , "ImageDehazer/Single-Results.html" , {'dehazedpdf':dehazedpdf , 'originalpdf': originalpdf ,  'dehazedphoto' : name , 'originalphoto' : original , 'psnr': psnr , 'ssim' :ssim , "method": method})
  
def get(request):
    originalphoto =  request.POST.get('originalphoto')
    dehazedphoto =  request.POST.get('dehazedphoto')
    method =  request.POST.get('method')
    psnr =  request.POST.get('psnr')
    ssim =  request.POST.get('ssim')
    imagename = request.POST.get('imagename')
    # originalphoto = originalphoto[1:]
    # dehazedphoto = dehazedphoto[2:]
    print(originalphoto , dehazedphoto , psnr , ssim , method)
    pdf = render_to_pdf('ImageDehazer/invoice.html',{'imagename' :imagename , 'dehazedphoto' : dehazedphoto , 'originalphoto' : originalphoto , 'psnr': psnr , 'ssim' :ssim , "method": method})
    return HttpResponse(pdf, content_type='application/pdf')

def datasetpdf(request):
    originalphoto =  request.POST.get('originalphoto')
    imagename = request.POST.get('imagename')
    dcpphoto = request.POST.get('dcpphoto')
    gfphoto = request.POST.get('gfphoto')

    params = PARAMS.objects.all()
    
    for i in params:
        if (i.imageID == imagename):
            localparam = i


    pdf = render_to_pdf('ImageDehazer/MultipleImageReport.html',{'imagename' :imagename ,"originalphoto" : originalphoto , "dcpphoto" :dcpphoto , "gfphoto" : gfphoto , "local" : localparam})
    return HttpResponse(pdf, content_type='application/pdf')


def qa(request):
    params = PARAMS.objects.all()
    print("fnkldslkvndkvn")
    return render(request, "ImageDehazer/Quantitative-Analysis.html" , {"params" : params})    