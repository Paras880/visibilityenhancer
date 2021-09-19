from ImageDehazer.models import Image
from io import BytesIO
from django.http import HttpResponse
from django.template.loader import get_template
import matplotlib.pyplot as plt
import base64
import random
import numpy as np

from xhtml2pdf import pisa

def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html  = template.render(context_dict)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), result)
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    return None

# def get_graph():
#     buffer = BytesIO()
#     plt.savefig(buffer , format ='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     print(image_png)
#     graph = base64.b64decode(image_png)
#     print("____________encoded______________________")
#     print(graph)
#     graph = graph.decode('ISO-8859-1')
#     print("____________encoded______________________")
#     print(graph)
#     buffer.close()
#     return graph

# def get_plot(x,y):
#     plt.switch_backend('AGG')
#     plt.figure(figsize=(5,2))
#     plt.title('PSNR Values')
#     plt.ylabel("Dehazing Methods")
#     plt.bar(x, y)
#     plt.xticks(rotation =45)
#     name = "/media/myimage/"+ str(random.randint(1,11111)) +".png"
#     plt.savefig("."+ name)
#     return name

def get_plot(x,y):

    X = ['Dcp','Gf','Cyclic-GAN','NovelIdea']

    X_axis = np.arange(len(X))

    plt.grid(which="both")
    plt.bar(X_axis - 0.2, x, 0.4, label = 'PSNR')
    plt.bar(X_axis + 0.2, y, 0.4, label = 'SSIM')
    
    plt.xticks(X_axis, X)
    plt.xlabel("Dehazing Methods")
    plt.title("Evaluation Parameters")
    plt.legend()

    name = "./media/myimage/"+ str(random.randint(1,11111)) +".png"
    plt.savefig(name)
    return name