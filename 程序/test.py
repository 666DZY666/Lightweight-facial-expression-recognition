import os
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import transforms as transform
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from models import vgg_prune
#from models import resnet_prune

use_cuda = False
cut_size = 46
fps1 = 0.0
fps2 = 0.0
v = 0.0000000001

pnet, rnet, onet = create_mtcnn_net(p_model_path="mtcnn_models/pnet.pt", r_model_path="mtcnn_models/rnet.pt", o_model_path="mtcnn_models/onet.pt", use_cuda=use_cuda)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size = 48, stride=2, threshold=[0.66, 0.7, 0.7], scale_factor=0.709)

#class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
class_names = ['just so so', 'just so so', 'just so so', 'good', 'common', 'just so so', 'common']

transform_test = transform.Compose([
    transform.TenCrop(cut_size),
    transform.Lambda(lambda crops: torch.stack([transform.ToTensor()(crop) for crop in crops])),
])

print('==>  ori_model ...')
net = vgg_prune.VGG()
checkpoint = torch.load('ori_models/fer_Pri_vgg16.pth')
net.load_state_dict(checkpoint['state_dict'])
#print('==>  ori_model ...')
#net = resnet_prune_1.resnet()
#checkpoint = torch.load('ori_models/fer_Pri_resnet164.pth')
#net.load_state_dict(checkpoint['state_dict'])
#print('==>  ori_model ...')
#net = resnet_prune.resnet18()
#checkpoint = torch.load('ori_models/fer_Pri_resnet18.pth')
#net.load_state_dict(checkpoint['state_dict'])
#print('==>  ori_model ...')
#net = resnet_prune.resnet101()
#checkpoint = torch.load('ori_models/fer_Pri_resnet101.pth')
#net.load_state_dict(checkpoint['state_dict'])

#print('==> prune_model ...')
#checkpoint1 = torch.load('prune_models/fer_Pri_vgg16_prune.pth')
#net = vgg_prune.VGG(cfg=checkpoint1['cfg'])
#checkpoint2 = torch.load('prune_models/fer_Pri_vgg16_refine.pth')
#net.load_state_dict(checkpoint2['state_dict'])
#print('==> prune_model ...')
#checkpoint1 = torch.load('prune_models/fer_Pri_resnet164_prune.pth')
#net = resnet_prune_1.resnet(cfg=checkpoint1['cfg'])
#checkpoint2 = torch.load('prune_models/fer_Pri_resnet164_refine.pth')
#net.load_state_dict(checkpoint2['state_dict'])

print(net)
net.eval()
if use_cuda:
    net.cuda()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,640) 
success, frame = cap.read()
#frame = cv2.imread('images/1.png')
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

while True:
    if success:
        success = 0

        #t = time.time()
        # *********************Face Detection**********************
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        #t1 = time.time() - t
        #fps1 = 1 / t1

        for i in range(bboxs.shape[0]):
            bbox = bboxs[i, :4]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            roi = frame[y1:y2, x1:x2]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
            roi = cv2.resize(roi, (49, 49))
            roi = roi[:, :, np.newaxis]
            roi = np.concatenate((roi, roi, roi), axis=2)
            roi = Image.fromarray(roi)
            inputs = transform_test(roi)
            inputs = Variable(inputs)
            if use_cuda:
                inputs = inputs.cuda()
            ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)

            t = time.time()
            # *********************FER***********************
            outputs = net(inputs)
            t2 = time.time() - t
            fps2 = 1 / (t2 + v)

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
            score = F.softmax(outputs_avg)
            _, predicted = torch.max(score.data, 0)
            label = str(class_names[int(predicted.cpu().numpy())])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #fps = 1 / (t1 + t2)
        #cv2.putText(frame, "FPS: " + str(fps)[:5], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #cv2.putText(frame, "FPS: " + str(fps1)[:5], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "FPS: " + str(fps2)[:5], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #for i in range(landmarks.shape[0]):
        #    landmarks_one = landmarks[i, :]
        #    landmarks_one = landmarks_one.reshape((5, 2))
        #    for j in range(5):
        #        # pylab.scatter(landmarks_one[j, 0], landmarks_one[j, 1], c='yellow', linewidths=0.1, marker='x', s=5)
        #        x3 = int(landmarks_one[j, 0])
        #        y3 = int(landmarks_one[j, 1])
        #        cv2.circle(frame, (x3, y3), 2, (0, 0, 255), 2)

        cv2.namedWindow("FER",0)
        cv2.imshow("FER",frame)  
        cv2.waitKey(1)

    success,frame = cap.read()
