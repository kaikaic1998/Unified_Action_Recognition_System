# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (640, 384))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose

    # [..., :2] = select the ----first two elements---- along the last dimension of a multidimensional NumPy array
    # [..., 2] = select the ----third element---- along the last dimension of a multidimensional NumPy array
    return result[..., :2], result[..., 2]


def main():
    args = parse_args()

    # args.video = video/breakdance.mp4
    # args.short_side = 480 (default set to 480)
    frame_paths, original_frames = frame_extraction(args.video, args.short_side)
    # frame_paths = all the temp stored images directories extracted from the video
    # original_frames = a list containing all images in form of list

    # Run human detectiom
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()
    # det_results contains all detections of all frames
    # det_results[i] contains all detections in one frame
    # det_results is list, length = num of frames, each sub-lists contains detections in one frame

    # run human keypoints from detected human with bbox
    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()
    # pose_results type = list
    # list lentgh = number of frames
    #
    # pose_results[i] length = num of detected human in one frame
    # pose_results[i] contains 2 dictionary, for all detected human
    #   one is 'bbox' : np.array of box coordinates
    #   one is 'keypoints' : np.array or key points (17 keypoints, same as COCO)
    #
    # pose_results[i][j] = bbox and kepoints dict for one human
    #
    # pose_results[i][j]['bbox] = np.array of bbox coordinates
    # pose_results[i][j]['keypoints'] = np.array of keypoints


    #----------------------------------------------------------------------------------------------
    #-------------------------------action recognition---------------------------------------------

    # num_frame = number of frames obtained from the video
    num_frame = len(frame_paths)
    
    # h = 480, w = 854
    h, w, _ = original_frames[0].shape

    # skeleton action recognition config file path
    # args.config = configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py
    config = mmcv.Config.fromfile(args.config)

    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    # skeleton action recognition checkpoint file/url
    # j.pth is stored in .cache folder
    # args.checkpoint = http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
    model = init_recognizer(config, args.checkpoint, args.device)

    # args.label_map = tools/data/label_map/nturgbd_120.txt
    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # config.data.test.pipeline:
    # [
    #     {'type': 'PreNormalize2D'}                                                , 
    #     {'type': 'GenSkeFeat', 'dataset': 'coco', 'feats': ['j']}                 , 
    #     {'type': 'UniformSample', 'clip_len': 100, 'num_clips': 10}               , 
    #     {'type': 'PoseDecode'}                                                    , 
    #     {'type': 'FormatGCNInput', 'num_person': 2}                               , 
    #     {'type': 'Collect', 'keys': ['keypoint', 'label'], 'meta_keys': []}       , 
    #     {'type': 'ToTensor', 'keys': ['keypoint']}
    # ]

    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type # GCN_flag = True
    
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)
    
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    # GCN_flag = True, this runs
    if GCN_flag:
        # We will keep at most `GCN_nperson` persons per frame.
        tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
        # tracking_inputs = all keypoints for all human in all frames

        # Run tracking from keypoints                        max_tracks = how many sets of keypoints (traking how many people)
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=2) 
        # len(keypoint) = max_tracks = number of tracked person
        #
        # keypoint[0] length = num of frames
        # 
        # keypoint[0][i] length = 17 (one set of keypoints)
        # keypoint[0][i] = a np.array containing 1 set of keypoints of one person, in one frame

        print('\nkeypoints shape: ', keypoint.shape) # (num of tracked person, 72, 17, 2)
        # print('keypoints[0] shape: ', keypoint[0].shape) # (72, 17, 2)
        # print('keypoints[0][0] shape: ', keypoint[0][0].shape) # (17, 2)
        # print('keypoints_score: ', keypoint_score.shape) # (2, 72, 17)

        # fake_anno type = dict
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
    else:
        num_person = max([len(x) for x in pose_results])
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                  dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    results = inference_recognizer(model, fake_anno)
    # results = a list of tuple with length of 4
    # each tuple = (action_label index, label confidence)
    # results[0] has the highest score, last tuple has the lowest score

    # results[0][0] = index of label with the highest score
    action_label = label_map[results[0][0]]
    # action_lable = name of the action

    #-------------------------------action recognition---------------------------------------------
    #----------------------------------------------------------------------------------------------

    # args.pose_config = demo/hrnet_w32_coco_256x192.py
    # args.pose_checkpoint = https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    # this is only for visualization
    # it is just how mmpose.apis works for visualizing

    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
        # cv2.imshow('', frame)
        # cv2.waitKey(10)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
