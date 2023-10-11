import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from threading import Thread
import time
from torchvision import transforms
from pathlib import Path
import os
import glob
import sys
import mmcv
from collections import deque
from collections import defaultdict


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

    results = inference_recognizer(GCN_model, fake_anno)
    action_label = label_map[results[0][0]]

    return action_label

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

def detect():
    # add yolov7 folder to path, to add it to the beginning of the module search path
    # otherwise, yolov7 module will causes "models" not found error
    sys.path.insert(0, 'yolov7')
    
    # weights = 'yolov7.pt'
    imgsz = 640

    # source = '0'
    # source = './video/palace.mp4'
    # source = './video/ntu_sample.avi'
    # source = './video/tennis.mp4'
    # source = './video/breakdance.mp4'
    source = './video/human_fall_2.mp4'

    # Initialize
    device = torch.device('cuda')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_path = './pretrained/yolov7-w6-pose.pt'

    weigths = torch.load(model_path, map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # model = TracedModel(model, device, imgsz)

    if half: # half = True
        model.half()  # to FP16

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    
    if source == '0':
        dataset = LoadWebcam(img_size=imgsz, stride=stride) # sources = '0' means webcam
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    tracker = BoTSORT(opt, frame_rate=30.0)

    #-------------------Action Recognition Model Initialization----------------------------
    num_total_frames = 0
    video = cv2.VideoCapture(source)
    while(True):
        ret, frame = video.read()
        if ret:
            num_total_frames += 1
        else:
            break
    video.release()

    num_input_to_GCN = 20

    config = mmcv.Config.fromfile('configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py')
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # args.checkpoint = http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
    #                 = stgcnpp_ntu120_xsub_j_6633e6c4.pth
    GCN_model = init_recognizer(config, '.cache/stgcnpp_ntu120_xset_hrnet.pth', device)
    # args.label_map = tools/data/label_map/nturgbd_120.txt
    # Load label_map
    label_map = [x.strip() for x in open('tools/data/label_map/nturgbd_120.txt').readlines()]

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(0, 0),
        # original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_input_to_GCN)
    #--------------------------------------------------------------------------------------

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once;

    start_time = time.time()

    # add one sub-list to input_toGCN for this one frame
    keypoints_dict = dict() # need this because output of id is not in number order
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
        img = img.half() if half else img.float()  # uint8 to fp16/32
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
            output, detections = output_to_keypoint_and_detections(output)
            # detections = output_to_keypoint_and_detections(output)
            # output and detections types are np.array

        # img: tensor, torch.Size([1, 3, 384, 640])
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        # # nimg: numpy array (384, 640, 3)
        h, w, _ = nimg.shape

        #------------------------------------------------------------------------- 
        #--------------------keypoints visualization------------------------------
        # for idx in range(output.shape[0]): # output.shape[0] = number of skeletons detected
        #     plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        for idx in range(detections.shape[0]): # output.shape[0] = number of skeletons detected
            plot_skeleton_kpts(nimg, detections[idx][5:].T, 3)
        #------------------------------------------------------------------------- 
        #--------------------keypoints visualization------------------------------

        #-------------------------------tracker operations--------------------------------------
        # image tpye for tracker should be numpy array, in format of (height, length, 3)
        online_targets = tracker.update(detections, nimg)

        online_tlwhs = []
        # online_ids = []
        online_scores = []
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
                online_tlwhs.append(tlwh)
                # online_ids.append(tid)
                online_ids[tid] += 1
                online_scores.append(t.score)

                if online_ids[tid] >= 3:
                    online_ids[tid] = 0
                    if tid in keypoints_dict: # if it is a new tracking
                        deque_len_of_this_id = len(keypoints_score_dict[tid])
                        print('deque_len_of_this_id: ', deque_len_of_this_id)

                        if deque_len_of_this_id >= num_input_to_GCN:
                            keypoints_dict[tid].popleft()
                            keypoints_score_dict[tid].popleft()

                            keypoints_dict[tid].append(keypoints)
                            keypoints_score_dict[tid].append(keypoints_score)

                            fake_anno['keypoint'] = np.array([keypoints_dict[tid]])
                            fake_anno['keypoint_score'] = np.array([keypoints_score_dict[tid]])
                            fake_anno['img_shape'] = (h, w)
                            action_label = GCN(fake_anno, GCN_model, label_map)
                        else:
                            keypoints_dict[tid].append(keypoints)
                            keypoints_score_dict[tid].append(keypoints_score)
                    else: # if the tracking is already exist
                        keypoints_dict[tid] = deque([keypoints])
                        keypoints_score_dict[tid] = deque([keypoints_score])

                # # action label for every tracked person
                action_label_dict[tid] = action_label

                label = f'{tid}, {action_label_dict[tid]}'
                plot_one_box(tlbr, nimg, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)

        cv2.imshow('', nimg)
        cv2.waitKey(1)  # 1 millisecond
        #---------------------------------------------------------------------

    end_time = time.time()
    execution_time = end_time - start_time
    print('time spent on inference: ', round(execution_time, 2))
    print('fps: ', round(num_total_frames/execution_time, 2))


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

    with torch.no_grad():
        detect()

