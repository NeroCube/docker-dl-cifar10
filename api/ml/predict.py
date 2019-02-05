from __future__ import print_function

import os
import torchvision.transforms as transforms

from ml.dpn import *
from ml.preact_resnet import *
from ml.densenet import *
from PIL import Image
from torch.autograd import Variable


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = DPN92()
net2 = PreActResNet18()
net3 = DenseNet121()


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

assert os.path.isdir('ml/checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./ml/checkpoint/ckpt.t7')
# net.load_state_dict(checkpoint['net'])
checkpoint = torch.load('./ml/checkpoint/dpn_ckpt.t7')
net.load_state_dict(checkpoint['net'])
checkpoint = torch.load('./ml/checkpoint/preact_resnet_ckpt.t7')
net2.load_state_dict(checkpoint['net'])
checkpoint = torch.load('./ml/checkpoint/ckpt.t7')
net3.load_state_dict(checkpoint['net'])



def predict(image):
    img_pil = Image.open(image.stream)
    image_tensor = preprocess(img_pil)
    print(image_tensor.size())
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)

    output = net(input)
    output2 = net2(input)
    output3 = net3(input)

    result = output.data.numpy()+output2.data.numpy()+output3.data.numpy()
    index = result.argmax()

    return classes[index]

# print(predict())