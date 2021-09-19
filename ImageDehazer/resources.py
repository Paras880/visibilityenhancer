from import_export import resources
from .models import PSNR
from .models import SSIM
from .models import PARAMS

class PSNRResource(resources.ModelResource):
    class Meta:
        model = PSNR

class SSIMResource(resources.ModelResource):
    class Meta:
        model = SSIM

class PARAMSResource(resources.ModelResource):
    class Meta:
        model = PARAMS