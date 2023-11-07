import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import time
import sys
import mmcv
from collections import deque
from collections import defaultdict

from yolov7.utils.general import check_img_size, non_max_suppression_kpt
from yolov7.utils.plots import plot_one_box, plot_skeleton_kpts

from tracker.mc_bot_sort import BoTSORT

from pyskl.apis import inference_recognizer, init_recognizer

from general_utils import LoadImages, LoadWebcam, output_to_keypoint_and_detections, args


def detect():
    sys.path.insert(0, 'yolov7')
    imgsz = 640
    source = opt.source

    # Initialize
    half = opt.device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_path = './pretrained/yolov7-w6-pose.pt'

    weigths = torch.load(model_path, map_location=opt.device)
    model = weigths['model']
    _ = model.float().eval()

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half: # half = True
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    tracker = BoTSORT(opt, frame_rate=30.0)

    if source == '0':
        dataset = LoadWebcam(img_size=imgsz, stride=stride)

        len_of_sliding_window = 20
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        num_total_frames = 0
        video = cv2.VideoCapture(source)
        while(True):
            ret, frame = video.read()
            if ret:
                num_total_frames += 1
            else:
                break
        video.release()

        len_of_sliding_window = int(num_total_frames/3) - 5

    #-------------------Action Recognition Model Initialization----------------------------
    config = mmcv.Config.fromfile(opt.stgcn_config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    config['model']['cls_head']['num_classes'] = 2

    GCN_model = init_recognizer(config, opt.new_stgcn_path, opt.device)
    
    # Load label_map
    label_map = [x.strip() for x in open('tools/data/label_map/new_label.txt').readlines()]

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(0, 0),
        start_index=0,
        modality='Pose',
        total_frames=len_of_sliding_window)
    #--------------------------------------------------------------------------------------

    # Run inference
    if opt.device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(opt.device).type_as(next(model.parameters())))  # run once;

    start_time = time.time()

    keypoints_dict = dict()
    keypoints_score_dict = dict()
    action_label_dict = dict()
    online_ids = defaultdict(int)
    action_label = ''
    
    count = 0
    for path, img, im0, vid_cap in dataset: # one dataset = one frame
        img = torch.from_numpy(img).to(opt.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
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
        h, w, _ = nimg.shape

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
                    if tid in keypoints_dict: # if it is a new tracking
                        deque_len_of_this_id = len(keypoints_score_dict[tid])

                        if deque_len_of_this_id >= len_of_sliding_window:
                            keypoints_dict[tid].popleft()
                            keypoints_score_dict[tid].popleft()

                            keypoints_dict[tid].append(keypoints)
                            keypoints_score_dict[tid].append(keypoints_score)

                            fake_anno['keypoint'] = np.array([keypoints_dict[tid]])
                            fake_anno['keypoint_score'] = np.array([keypoints_score_dict[tid]])
                            fake_anno['img_shape'] = (h, w)
                            results = inference_recognizer(GCN_model, fake_anno)
                            action_label = label_map[results[0][0]]
                        else:
                            keypoints_dict[tid].append(keypoints)
                            keypoints_score_dict[tid].append(keypoints_score)
                    else: # if the tracking is already exist
                        keypoints_dict[tid] = deque([keypoints])
                        keypoints_score_dict[tid] = deque([keypoints_score])

                # label for every tracked person
                action_label_dict[tid] = action_label

                label = f'{tid}, {action_label_dict[tid]}'
                plot_one_box(tlbr, nimg, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)

        cv2.imshow('', nimg)
        cv2.waitKey(1)  # 1 millisecond
        count += 1

    end_time = time.time()
    execution_time = end_time - start_time
    print('time spent on inference: ', round(execution_time, 2))
    if source != '0':
        print('fps: ', round(num_total_frames/execution_time, 2))


if __name__ == '__main__':
    opt = args()
    opt.jde = False
    opt.ablation = False

    with torch.no_grad():
        detect()

