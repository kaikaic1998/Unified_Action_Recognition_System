from mmcv import Config
import argparse

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from threading import Thread
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
import os
import glob
import sys
import mmcv
from collections import deque
from collections import defaultdict
import pathlib

from pyskl.models.recognizers.recognizergcn import RecognizerGCN

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, non_max_suppression_kpt
from yolov7.utils.general import xywh2xyxy, xyxy2xywh
from yolov7.utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from yolov7.utils.torch_utils import TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

from pyskl.apis import inference_recognizer, init_recognizer

# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print(f'proccessing video frame {self.frame} out of {self.nframes}')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
    
        # return path, img, self.cap
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

#------------------------------------------------------------------------------------------------
def GCN(fake_anno, GCN_model, label_map):

    # config = mmcv.Config.fromfile('configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py')
    # config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # # args.checkpoint = http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
    # GCN_model = init_recognizer(config, '.cache/j_6633e6c4.pth', device)
    # # args.label_map = tools/data/label_map/nturgbd_120.txt
    # # Load label_map
    # label_map = [x.strip() for x in open('tools/data/label_map/nturgbd_120.txt').readlines()]

    # fake_anno = dict(
    #     frame_dir='',
    #     label=-1,
    #     img_shape=(h, w),
    #     original_shape=(h, w),
    #     start_index=0,
    #     modality='Pose',
    #     total_frames=num_frame)

    results, all_result = inference_recognizer(GCN_model, fake_anno)
    # print('result: ', all_result)
    # print('results len: ', len(all_result))
    # print('results[0][0]: ', all_result[0][0])
    # print('results[0][0]: ', results[0][0])

    all_scores = []
    for i in range(len(all_result)):
        all_scores.append(all_result[i][1])
    # print('\n', all_scores)
    # # print('results[0][0]: ', results[0][0])
    # print('result tuple: ', all_result[results[0][0]])

    all_scores = torch.tensor([all_scores], requires_grad=True)
    action_label = label_map[results[0][0]]

    return action_label, all_scores

#--------------------------------------------------------------------------------------------
def output_to_keypoint_and_detections(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    detections = []

    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6] # all detected boxes, format: tensor([ [box_coordinates_xyxy, confidence, class] x #of detections ]) 
        # detections.append(o.cpu().numpy())
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            # box = the bounding box in form of x1 y1 x2 y2,  where xy1=top-left, xy2=bottom-right
            # xyxy2xywh() converts nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
            # print('keypoints: ', list(kpts.detach().cpu().numpy()[index]))
            # print('index: ', index)
            # cls = sum(list(kpts.detach().cpu().numpy()[index]))/len(list(kpts.detach().cpu().numpy()[index]))
            # detections.append([*box, conf, cls])
            detections.append([*box, conf, *(list(kpts.detach().cpu().numpy()[index]))])


    return np.array(targets), np.array(detections)
#---------------------------------------------------------------------------------------------

class video_to_keypoint_dataset(Dataset):
    def __init__(self, path, device):
        self.device = device
        self.path = path
        self.paths = list(pathlib.Path(path).glob("*/*.*"))
        self.classes, self.class_to_idx = self.find_class(path)

        self.all_keypoints, self.all_keypoints_score = self.get_all_keypoints_from_all_video()

    def find_class(self, class_path):
            class_names = os.listdir(class_path)
            class_to_idx = {name: i for i, name in enumerate(class_names)}
            return class_names, class_to_idx
    
    def __getitem__(self, index):
        class_name = os.path.basename(os.path.dirname(self.paths[index]))
        class_index = self.class_to_idx[class_name]
        
        # return (skeleton of this video, class_of_this_index)
        return (self.all_keypoints[index], self.all_keypoints_score[index], class_index, class_name)
    
    def __len__(self):
        return len(self.paths)

    def get_all_keypoints_from_all_video(self):
        all_keypoints = []
        all_keypoints_score = []

        with torch.no_grad():
            model, tracker, stride, imgsz = self.init_track_pose(self.device)

            for video_path in self.paths:
                print('Generating keypoints data for: ', video_path)
                keypoints_of_one_video, keypoints_score_of_one_video = self.keypoints_of_one_video(video_path, self.device, model, tracker, stride, imgsz)
                all_keypoints.append(keypoints_of_one_video)
                all_keypoints_score.append(keypoints_score_of_one_video)

        return all_keypoints, all_keypoints_score
   
    def init_track_pose(self, device):
        sys.path.insert(0, 'yolov7')
        
        # weights = 'yolov7.pt'
        imgsz = 640

        # Load model
        model_path = './pretrained/yolov7-w6-pose.pt'

        weigths = torch.load(model_path, map_location=device)
        model = weigths['model']
        _ = model.float().eval()

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        # model = TracedModel(model, device, imgsz)

        if (device.type != 'cpu'): # half = True
            model.half()  # to FP16

        # Set Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference

        tracker = BoTSORT(opt, frame_rate=30.0)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once;

        return model, tracker, stride, imgsz

    def keypoints_of_one_video(self, source, device, model, tracker, stride, imgsz):
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        num_total_frames = dataset.nframes
        num_input_to_GCN = int(num_total_frames/3) - 6

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

        id_list = []
        keypoints_dict = dict()
        keypoints_score_dict = dict()
        action_label_dict = dict()
        online_ids = defaultdict(int)
        action_label = ''

        for path, img, im0, vid_cap in dataset: # one dataset = one frame
            # img: numpy array (384, 640, 3)
            #--------------------------------------------------------------
            # change img type from numpy to tensor
            # so can feed img to GPU, to speed up inference speed
            img = torch.from_numpy(img).to(device)
            img = img.half() if (device.type != 'cpu') else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            img = img.permute(0,3,1,2)
            with torch.no_grad():
                output, _ = model(img)
            #--------------------------------------------------------------
            # img: tensor, torch.Size([1, 3, 384, 640])

            # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                # pred = model(img, augment=opt.augment)[0] # opt.augment value is False here

            # Apply NMS
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

            with torch.no_grad():
                # output = all keypoints in 1 frame
                # detections = all bbox in 1 frame, in form of [[*box, conf, cls], x num_bbox]
                targets, detections = output_to_keypoint_and_detections(output)

            # img: tensor, torch.Size([1, 3, 384, 640])
            nimg = img[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            # # nimg: numpy array (384, 640, 3)

            # image tpye for tracker should be numpy array, in format of (height, length, 3)
            online_targets = tracker.update(detections, nimg)

            # online_tlwhs = []
            # online_ids = []
            # online_scores = []
            # online_cls = []

            for t in online_targets:
                tlwh = t.tlwh # used for filtering out small boxes
                tlbr = t.tlbr # bbox coordinates
                tid = t.track_id # a number id for each tracked person, tpye: int

                #-----------------keypoints and scores for one tracked person in this one frame--------------
                keypoints = []
                keypoints_score = []

                steps = 3
                num_keypoints = len(t.cls) // steps

                for i in range(num_keypoints):
                    x_coord, y_coord = t.cls[steps * i], t.cls[steps * i + 1]
                    keypoints.append([x_coord, y_coord])
                    keypoints_score.append(t.cls[steps * i + 2])
                #---------------------------------------------------------------------------------------------

                if tlwh[2] * tlwh[3] > opt.min_box_area: # filter out small boxes
                    # online_tlwhs.append(tlwh)
                    # online_ids.append(tid)
                    online_ids[tid] += 1
                    # online_scores.append(t.score)

                    if online_ids[tid] >= 3:
                        online_ids[tid] = 0
                        if tid in keypoints_dict: # if it is an existing tracking id
                            deque_len_of_this_id = len(keypoints_score_dict[tid])
                            # print('deque_len_of_this_id: ', deque_len_of_this_id)

                            if deque_len_of_this_id >= num_input_to_GCN:
                                keypoints_dict[tid].popleft()
                                keypoints_score_dict[tid].popleft()

                                keypoints_dict[tid].append(keypoints)
                                keypoints_score_dict[tid].append(keypoints_score)
                                # fake_anno['keypoint'] = np.array([keypoints_dict[tid]])
                                # fake_anno['keypoint_score'] = np.array([keypoints_score_dict[tid]])
                                # fake_anno['img_shape'] = (h, w)
                                # action_label = GCN(fake_anno, GCN_model, label_map)
                            else:
                                keypoints_dict[tid].append(keypoints)
                                keypoints_score_dict[tid].append(keypoints_score)
                        else: # if the tracking id is new
                            id_list.append(tid)
                            keypoints_dict[tid] = deque([keypoints])
                            keypoints_score_dict[tid] = deque([keypoints_score])

                    # # action label for every tracked person
                    # action_label_dict[tid] = action_label

                    label = f'{tid}'
                    plot_one_box(tlbr, nimg, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            # cv2.imshow('', nimg)
            # cv2.waitKey(1)  # 1 millisecond
            #---------------------------------------------------------------------

        keypoints_from_all_tracking = []
        keypoints_score_from_all_tracking = []
        for id in id_list:
            keypoints_from_all_tracking.append(list(keypoints_dict[id]))
            keypoints_score_from_all_tracking.append(list(keypoints_score_dict[id]))
        
        return np.array(keypoints_from_all_tracking), np.array(keypoints_score_from_all_tracking)

# class Custom_STGCN_Model(RecognizerGCN):
class Custom_STGCN_Model(nn.Module):
    def __init__(self, config, device, label_map):
        super(Custom_STGCN_Model, self).__init__()
        self.label_map = label_map

        self.model = init_recognizer(config, '.cache/stgcnpp_ntu120_xset_hrnet.pth', device)

        for param in self.model.parameters():
            param.requires_grad = False

        # self.model.cls_head.fc_cls = nn.Linear(self.model.cls_head.in_c, 2)
        self.model.cls_head.fc_cls = nn.Sequential(
            nn.Linear(self.model.cls_head.in_c, 120),
            nn.ReLU(),
            nn.Linear(120, 2),
        )
        # torch.nn.init.xavier_uniform(GCN_model.cls_head.fc_cls.weight)

        for param in self.model.cls_head.parameters():
            param.requires_grad = True

        self.model = self.model.to(device)

        print('model head: ', self.model.cls_head)

    def forward(self, fake_anno):
        results, all_result = inference_recognizer(self.model, fake_anno)

        all_scores = []
        for i in range(len(all_result)):
            all_scores.append(all_result[i][1])
        # print('\n', all_scores)
        # # print('results[0][0]: ', results[0][0])
        # print('result tuple: ', all_result[results[0][0]])

        all_scores = torch.tensor([all_scores], requires_grad=True)
        action_label = label_map[results[0][0]]

        for param in self.model.cls_head.parameters():
            print('param: ', param)

        return action_label, all_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--name', default='exp', help='save results to project/name')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    #################################################################################################################
    #################################################################################################################
    device = torch.device('cuda')

    dataset = video_to_keypoint_dataset(path='./dataset/', device=device)
    # print('\ndataset.class_to_idx: ', dataset.class_to_idx)
    # print('dataset.classes:, ', dataset.classes)
    # print('dataset.path: ', dataset.path)
    # print('dataset.paths: ', dataset.paths)
    # print('dataset len: ', len(dataset))
    # print('----------------------------------------------------\n')

    config = mmcv.Config.fromfile('configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py')
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    #------
    # GCN_model = init_recognizer(config, '.cache/stgcnpp_ntu120_xset_hrnet.pth', device)
    # print(GCN_model.cls_head)
    # (cls_head): GCNHead(
    #     (loss_cls): CrossEntropyLoss()
    #     (fc_cls): Linear(in_features=256, out_features=120, bias=True)
    #   )

    # print(GCN_model.cls_head)
    # print(GCN_model.cls_head.__dict__)
    # print(GCN_model.cls_head.in_c) # 256
    # print(GCN_model.cls_head.num_classes) # 120
    # print(GCN_model.cls_head.fc_cls) # Linear(in_features=256, out_features=120, bias=True)

    # # True = not freeze parameters
    # # False = freeze parameters
    # for param in GCN_model.parameters():
    #     param.requires_grad = False

    # GCN_model.cls_head.fc_cls = nn.Linear(GCN_model.cls_head.in_c, 2)
    # # torch.nn.init.xavier_uniform(GCN_model.cls_head.fc_cls.weight)
    # GCN_model = GCN_model.to(device) # need to send to cuda again, because above line somehow makes model to CPU again
    # print('GCN_model.cls_head: ', GCN_model.cls_head)
    # # # GCN_model.cls_head:  GCNHead(
    # # (loss_cls): CrossEntropyLoss()
    # # (fc_cls): Linear(in_features=256, out_features=2, bias=True)
    # # )
    # # for param in GCN_model.cls_head.fc_cls.parameters():
    # for param in GCN_model.cls_head.parameters():
    #     param.requires_grad = True

    # label_map = [x.strip() for x in open('tools/data/label_map/nturgbd_120.txt').readlines()]
    label_map = [x.strip() for x in open('tools/data/label_map/new2.txt').readlines()]
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(0, 0),
        # original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=0)

    GCN_model = Custom_STGCN_Model(config, device, label_map)
    #---------------------------for training part------------------------------

    # if config.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True

    # optimizer
    # optimizer = torch.optim.SGD(GCN_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer = torch.optim.SGD(GCN_model.parameters(), lr=0.01)

    # loss function
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    loss_list = []
    # train
    # no need Pytorch Dataloader, because can only feed one set of keypoints at a time to GCN
    for index, data in enumerate(dataset):
        pred_keypoints, pred_keypoints_score, pred_class, pred_class_name = data[0], data[1], data[2], data[3]

        pred_class = torch.tensor([pred_class], dtype=torch.int64)

        # print('index: ', index)
        # print('keypoints shape: ', pred_keypoints.shape)
        # print('keypoints_score shape: ', pred_keypoints_score.shape)
        # print('class: ', pred_class)
        # print('class name: ', pred_class_name)

        fake_anno['keypoint'] = pred_keypoints
        fake_anno['keypoint_score'] = pred_keypoints_score
        fake_anno['img_shape'] = (384, 640)
        fake_anno['total_frames'] = pred_keypoints_score.shape[1]

        GCN_model.train()
        optimizer.zero_grad()

        # pred_label, pred_scores = GCN(fake_anno, GCN_model, label_map)
        pred_label, pred_scores = GCN_model(fake_anno)

        log_pred_scores = torch.log(pred_scores)

        print('label: ', pred_label)
        print('class: ', pred_class)

        loss = criterion(log_pred_scores, pred_class) # criterion(prediction, ground truth)

        loss.backward()
        optimizer.step()
        print('loss: ', loss)

        loss_list.append(loss.item())
        
        # for param in GCN_model.cls_head.parameters():
        #     print('param: ', param)



    print('\nloss list: ', loss_list)


