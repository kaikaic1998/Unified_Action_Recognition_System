import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os
import glob
import sys
import time
from threading import Thread
from collections import deque
from collections import defaultdict
import pathlib
import random
from matplotlib import pyplot as plt
import matplotlib

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression_kpt
from yolov7.utils.plots import plot_one_box, plot_skeleton_kpts

from tracker.mc_bot_sort import BoTSORT

from pyskl.apis import inference_recognizer, train_recognizer

matplotlib.use('TkAgg')


# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class LoadImages:  # video and images
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


class LoadWebcam:  # webcam
    def __init__(self, img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        self.imgs = [None]

        self.cap = cv2.VideoCapture(0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) % 100

        _, self.imgs[0] = self.cap.read()  # guarantee first frame

        thread = Thread(target=self.update, args=([self.cap]), daemon=True)
        print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
        thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, cap):
        index = 0

        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        # Stack
        img = np.stack(img, 0)

        return '0', img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class video_to_keypoint_dataset(Dataset):
    def __init__(self, path, device, yolo_model_path, opt, show_img, imgsz=640):
        self.device = device
        self.imgsz = imgsz
        self.yolo_model_path = yolo_model_path
        self.show_img = show_img
        self.opt = opt
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
        
        return (self.all_keypoints[index], self.all_keypoints_score[index], class_index, class_name)
    
    def __len__(self):
        return len(self.paths)

    def get_all_keypoints_from_all_video(self):
        all_keypoints = []
        all_keypoints_score = []

        with torch.no_grad():
            model, tracker, stride, imgsz = self.init_track_pose(self.device, self.imgsz, self.yolo_model_path, self.opt)

        # get keypoints info from all videos
        for video_path in self.paths:
            print('Generating keypoints data for: ', video_path)
            keypoints_of_one_video, keypoints_score_of_one_video = self.keypoints_of_one_video(video_path, self.device, model, tracker, stride, imgsz, self.show_img, self.opt)
            all_keypoints.append(keypoints_of_one_video)
            all_keypoints_score.append(keypoints_score_of_one_video)

        return all_keypoints, all_keypoints_score
   
    def init_track_pose(self, device, imgsz, model_path, opt):
        sys.path.insert(0, 'yolov7')

        weigths = torch.load(model_path, map_location=device)
        model = weigths['model']
        _ = model.float().eval()

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size


        if (device.type != 'cpu'): # half = True
            model.half()  # to FP16

        cudnn.benchmark = True  # set True to speed up constant image size inference

        tracker = BoTSORT(opt, frame_rate=30.0)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once;

        return model, tracker, stride, imgsz

    def keypoints_of_one_video(self, source, device, model, tracker, stride, imgsz, show_img, opt):
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        len_of_sliding_window = int(dataset.nframes/3) - 5

        if show_img:
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

        id_list = []
        keypoints_dict = dict()
        keypoints_score_dict = dict()
        online_ids = defaultdict(int)

        for path, img, im0, vid_cap in dataset: # one dataset = one frame
            img = torch.from_numpy(img).to(device)
            img = img.half() if (device.type != 'cpu') else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            img = img.permute(0,3,1,2)
            with torch.no_grad():
                output, _ = model(img)

            # Apply NMS
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

            with torch.no_grad():
                detections = output_to_keypoint_and_detections(output)

            nimg = img[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)

            if show_img:
                for idx in range(detections.shape[0]):
                    plot_skeleton_kpts(nimg, detections[idx][5:].T, 3)

            online_targets = tracker.update(detections, nimg)

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
                    online_ids[tid] += 1

                    if online_ids[tid] >= 3:
                        online_ids[tid] = 0
                        if tid in keypoints_dict: # if it is an existing tracking id
                            deque_len_of_this_id = len(keypoints_score_dict[tid])

                            if deque_len_of_this_id >= len_of_sliding_window:
                                keypoints_dict[tid].popleft()
                                keypoints_score_dict[tid].popleft()

                                keypoints_dict[tid].append(keypoints)
                                keypoints_score_dict[tid].append(keypoints_score)
                            else:
                                keypoints_dict[tid].append(keypoints)
                                keypoints_score_dict[tid].append(keypoints_score)
                        else: # if the tracking id is new
                            id_list.append(tid)
                            keypoints_dict[tid] = deque([keypoints])
                            keypoints_score_dict[tid] = deque([keypoints_score])
                    
                    if show_img:
                        label = f'{tid}'
                        plot_one_box(tlbr, nimg, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            
            if show_img:
                cv2.imshow('', nimg)
                cv2.waitKey(1)  # 1 millisecond

        keypoints_from_all_tracking = []
        keypoints_score_from_all_tracking = []
        for id in id_list:
            keypoints_from_all_tracking.append(list(keypoints_dict[id]))
            keypoints_score_from_all_tracking.append(list(keypoints_score_dict[id]))
        
        return np.array(keypoints_from_all_tracking), np.array(keypoints_score_from_all_tracking)


def args():
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--yolo-model-path', default='pretrained/yolov7-w6-pose.pt', help='path of the pretrained YOLO model')
    parser.add_argument('--show-img', default=False, help='Show tracking and keypoints when training')
    parser.add_argument('--stgcn-config', default='configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py', help='config for pretrained STGCN model')
    parser.add_argument('--stgcn-path', default='pretrained/stgcnpp_ntu120_xset_hrnet.pth', help='pretrained STGCN model path')
    parser.add_argument('--new-stgcn-path', default='pretrained/new_model.pth', help='newly fine-tuned STGCN model path')
    parser.add_argument('--save-model', default=False, help='save newly fine-tuned STGCN model')
    parser.add_argument('--lr', default=0.01, help='training learning rate')
    parser.add_argument('--epoch', default=30, help='training epoch')
    parser.add_argument('--source', default='./video/fall.mp4', help="demo video, '0' for webcam")
    parser.add_argument('--label-path', default='label_map/new_label.txt', help='labels for inference')

    # detector args
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
    
    return parser.parse_args()


def output_to_keypoint_and_detections(output):
    detections = []

    for o in output:
        kpts = o[:,6:]
        o = o[:,:6] # all detected boxes, format: tensor([ [box_coordinates_xyxy, confidence, class] x #of detections ]) 
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            detections.append([*box, conf, *(list(kpts.detach().cpu().numpy()[index]))])
    return np.array(detections)


def train_model(GCN_model, device, train_dataset, fake_anno, optimizer):
    random_index = list(range(len(train_dataset)))
    random.shuffle(random_index)

    loss_sublist = []

    for i in random_index:
        data = train_dataset[i]

        pred_keypoints, pred_keypoints_score, gt_class, gt_class_name = data[0], data[1], data[2], data[3]

        gt_class = torch.tensor([gt_class], dtype=torch.int64).to(device)

        fake_anno['keypoint'] = pred_keypoints
        fake_anno['keypoint_score'] = pred_keypoints_score
        fake_anno['img_shape'] = (384, 640)
        fake_anno['total_frames'] = pred_keypoints_score.shape[1]

        GCN_model.cls_head.fc_cls.train()
        optimizer.zero_grad()

        loss = train_recognizer(GCN_model, fake_anno, gt_class)

        loss.backward()
        optimizer.step()

        loss_sublist.append(loss.item())
    
    return loss_sublist


def val_model(GCN_model, val_dataset, fake_anno):
    len_dataset = len(val_dataset)

    correct = 0

    for data in val_dataset:
        pred_keypoints, pred_keypoints_score, gt_class, gt_class_name = data[0], data[1], data[2], data[3]

        fake_anno['keypoint'] = pred_keypoints
        fake_anno['keypoint_score'] = pred_keypoints_score
        fake_anno['img_shape'] = (384, 640)
        fake_anno['total_frames'] = pred_keypoints_score.shape[1]

        GCN_model.eval()
        
        results = inference_recognizer(GCN_model, fake_anno)

        correct += (gt_class == results[0][0])

    return correct / len_dataset


def plot_result(loss, accuracy):
    plt.plot(loss, '-o', accuracy, '-o')
    plt.grid()
    plt.show()
