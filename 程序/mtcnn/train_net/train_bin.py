from dface.core.image_reader import TrainImageReader
import datetime
import os
from dface.core.bin_model import PNet,RNet,ONet,LossFn
import torch
from torch.autograd import Variable
import dface.core.image_tools as image_tools
import bin_util
import matplotlib.pyplot as plt

def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    #we only need the detection which >= 0
    mask = torch.ge(gt_cls,0)
    #get valid element
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))


def train_pnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=use_cuda)
    net.train()
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40, 45], gamma=0.1)

    # define the binarization operator
    bin_op = bin_util.BinOp(net)

    train_data=TrainImageReader(imdb,12,batch_size,shuffle=True)

    accuracy_avg_list = []
    cls_loss_avg_list = []
    bbox_loss_avg_list = []
    all_loss_avg_list = []
    x1 = range(0, 50)
    x2 = range(0, 50)
    x3 = range(0, 50)
    x4 = range(0, 50)

    for cur_epoch in range(0, end_epoch):
        scheduler.step()

        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        # landmark_loss_list=[]
        all_loss_list = []

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            # gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            # ！！！权重（参数）二值化 ！！！
            bin_op.binarization()   #含缩放因子

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                # gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)#含缩放因子
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = str(accuracy.data.tolist())
                show2 = str(cls_loss.data.tolist())
                show3 = str(box_offset_loss.data.tolist())
                show5 = str(all_loss.data.tolist())

                print("%s : Epoch: %d, Step: %d, accuracy: %s, cls loss: %s, bbox loss: %s, all_loss: %s, lr:%.6f "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,scheduler.get_lr()[0]))
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(box_offset_loss)
                all_loss_list.append(all_loss)

            optimizer.zero_grad()
            all_loss.backward()             #计算梯度（指定loss）

            bin_op.restore()
            bin_op.updateBinaryGradWeight() #加入缩放因子

            optimizer.step()                #使用梯度，更新参数（指定optimizer）

        accuracy_avg = torch.mean(torch.stack(accuracy_list, dim=0))
        accuracy_avg_list.append(accuracy_avg)
        cls_loss_avg = torch.mean(torch.stack(cls_loss_list, dim=0))
        cls_loss_avg_list.append(cls_loss_avg)
        bbox_loss_avg = torch.mean(torch.stack(bbox_loss_list, dim=0))
        bbox_loss_avg_list.append(bbox_loss_avg)
        # landmark_loss_avg = torch.mean(torch.cat(landmark_loss_list))
        all_loss_avg = torch.mean(torch.stack(all_loss_list, dim=0))
        all_loss_avg_list.append(all_loss_avg)

        show6 = str(accuracy_avg.data.tolist())
        show7 = str(cls_loss_avg.data.tolist())
        show8 = str(bbox_loss_avg.data.tolist())
        show10 = str(all_loss_avg.data.tolist())

        print("Epoch: %d, accuracy: %s, cls loss: %s, bbox loss: %s, all_loss: %s" % (cur_epoch, show6, show7, show8, show10))

        #net = net.module
        torch.save(net.module.state_dict(), os.path.join(model_store_path,"3_bin_pnet_epoch_%d.pt" % cur_epoch))
        torch.save(net.module, os.path.join(model_store_path,"3_bin_pnet_epoch_model_%d.pkl" % cur_epoch))

    y1 = accuracy_avg_list
    y2 = cls_loss_avg_list
    y3 = bbox_loss_avg_list
    y4 = all_loss_avg_list

    plt.subplot(1, 4, 1)
    plt.title('Bin-P-Net')
    plt.plot(x1, y1, 'o-')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.subplot(1, 4, 2)
    plt.plot(x2, y2, 'o-')
    plt.xlabel('Epoches')
    plt.ylabel('Cls_loss')
    plt.subplot(1, 4, 3)
    plt.plot(x3, y3, 'o-')
    plt.xlabel('Epoches')
    plt.ylabel('Bbox_loss')
    plt.subplot(1, 4, 4)
    plt.plot(x4, y4, 'o-')
    plt.xlabel('Epoches')
    plt.ylabel('All_loss')
    plt.show()
    plt.savefig("Bin-accuracy-epoches.jpg")

def train_rnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = RNet(is_train=True, use_cuda=use_cuda)
    net.train()
    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,24,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        landmark_loss_list=[]

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.tolist()[0]
                show2 = cls_loss.data.tolist()[0]
                show3 = box_offset_loss.data.tolist()[0]
                # show4 = landmark_loss.data.tolist()[0]
                show5 = all_loss.data.tolist()[0]

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, base_lr))
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(box_offset_loss)
                # landmark_loss_list.append(landmark_loss)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()


        accuracy_avg = torch.mean(torch.cat(accuracy_list))
        cls_loss_avg = torch.mean(torch.cat(cls_loss_list))
        bbox_loss_avg = torch.mean(torch.cat(bbox_loss_list))
        # landmark_loss_avg = torch.mean(torch.cat(landmark_loss_list))

        show6 = accuracy_avg.data.tolist()[0]
        show7 = cls_loss_avg.data.tolist()[0]
        show8 = bbox_loss_avg.data.tolist()[0]
        # show9 = landmark_loss_avg.data.tolist()[0]

        print("Epoch: %d, accuracy: %s, cls loss: %s, bbox loss: %s" % (cur_epoch, show6, show7, show8))
        torch.save(net.state_dict(), os.path.join(model_store_path,"rnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"rnet_epoch_model_%d.pkl" % cur_epoch))


def train_onet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = ONet(is_train=True)
    net.train()
    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,48,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        landmark_loss_list=[]

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*0.8+box_offset_loss*0.6+landmark_loss*1.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.tolist()[0]
                show2 = cls_loss.data.tolist()[0]
                show3 = box_offset_loss.data.tolist()[0]
                show4 = landmark_loss.data.tolist()[0]
                show5 = all_loss.data.tolist()[0]

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, landmark loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show4,show5,base_lr))
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(box_offset_loss)
                landmark_loss_list.append(landmark_loss)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()


        accuracy_avg = torch.mean(torch.cat(accuracy_list))
        cls_loss_avg = torch.mean(torch.cat(cls_loss_list))
        bbox_loss_avg = torch.mean(torch.cat(bbox_loss_list))
        landmark_loss_avg = torch.mean(torch.cat(landmark_loss_list))

        show6 = accuracy_avg.data.tolist()[0]
        show7 = cls_loss_avg.data.tolist()[0]
        show8 = bbox_loss_avg.data.tolist()[0]
        show9 = landmark_loss_avg.data.tolist()[0]

        print("Epoch: %d, accuracy: %s, cls loss: %s, bbox loss: %s, landmark loss: %s " % (cur_epoch, show6, show7, show8, show9))
        torch.save(net.state_dict(), os.path.join(model_store_path,"onet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))

