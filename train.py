import torch
import torch.nn as nn
import numpy as np
import mmcv

from pyskl.apis import init_recognizer

from general_utils import video_to_keypoint_dataset, args, train_model, val_model, plot_result


if __name__ == '__main__':
    opt = args()
    opt.jde = False
    opt.ablation = False

    train_dataset = video_to_keypoint_dataset(path='./dataset_train/', device=opt.device, yolo_model_path=opt.yolo_model_path, opt=opt, show_img=opt.show_img)
    val_dataset = video_to_keypoint_dataset(path='./dataset_val/', device=opt.device, yolo_model_path=opt.yolo_model_path, opt=opt, show_img=opt.show_img)

    # create new class labels from training, for reference
    if opt.create_new_label:
        f = open('tools/data/label_map/new_label.txt', 'w')
        for class_name in train_dataset.classes:
            text = str(class_name) + '\n'
            f.write(text)
        f.close()
        print('\nCreated new class labels from training dataset: \ntools/data/label_map/new_label.txt\n')

    #--------------initialize recognizer---------------
    config = mmcv.Config.fromfile(opt.stgcn_config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    config.data.test.pipeline[2].pop('num_clips')
    
    # load original pretrained STGCN model
    GCN_model = init_recognizer(config, opt.stgcn_path, opt.device)

    for param in GCN_model.parameters():
        param.requires_grad = False

    GCN_model.cls_head.fc_cls = nn.Linear(GCN_model.cls_head.in_c, 2)

    for param in GCN_model.cls_head.fc_cls.parameters():
        param.requires_grad = True

    GCN_model = GCN_model.to(opt.device)

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(0, 0),
        start_index=0,
        modality='Pose',
        total_frames=0)
    #-----------------------------------------------------

    if config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # optimizer
    optimizer = torch.optim.Adam(GCN_model.parameters(), lr=opt.lr)

    loss_list = []
    accuracy_list = []

    epochs = opt.epoch

    # start training
    for epoch in range(epochs):
        loss_sublist = train_model(GCN_model, opt.device, train_dataset, fake_anno, optimizer)
        loss_list.append(np.mean(loss_sublist))

        accuracy = val_model(GCN_model, val_dataset, fake_anno)
        accuracy_list.append(accuracy)

        print('Epoch ', epoch+1, ' done.')

    # plot training result
    plot_result(loss_list, accuracy_list)

    if opt.save_model:
        torch.save(GCN_model.state_dict(), 'pretrained/new_model.pth')
        print('\nSaved newly fine-tuned model: pretrained/new_model.pth\n')