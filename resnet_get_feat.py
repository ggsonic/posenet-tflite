import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

model = models.resnet50(pretrained=True)
model.fc=nn.Sequential()
#layer = model.avgpool
model.eval()

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name).convert('RGB')
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    v=model(t_img)
    return v

v1=get_vector('ocr/3.png')
v2=get_vector('ocr/2.png')
v3=get_vector('ocr/4.png')
v4=get_vector('ocr/5.png')
v5=get_vector('ocr/6.png')
v6=get_vector('ocr/7.png')
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
cos_sim = cos(v1[0], v2[0])
print(cos_sim.data.numpy()+0)
cos_sim = cos(v1[0], v3[0])
print(cos_sim.data.numpy()+0)
cos_sim = cos(v1[0], v4[0])
print(cos_sim.data.numpy()+0)
cos_sim = cos(v1[0], v5[0])
print(cos_sim.data.numpy()+0)
cos_sim = cos(v1[0], v6[0])
print(cos_sim.data.numpy()+0)
cos_sim = cos(v3[0], v4[0])
print(cos_sim.data.numpy()+0)
    



