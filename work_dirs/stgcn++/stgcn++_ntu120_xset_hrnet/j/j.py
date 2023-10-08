model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=120, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'tools/data/ntu120_hrnet.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='PoseDataset',
            ann_file='tools/data/ntu120_hrnet.pkl',
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(type='GenSkeFeat', dataset='coco', feats=['j']),
                dict(type='UniformSample', clip_len=100),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='xset_train')),
    val=dict(
        type='PoseDataset',
        ann_file='tools/data/ntu120_hrnet.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='xset_val'),
    test=dict(
        type='PoseDataset',
        ann_file='tools/data/ntu120_hrnet.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=10),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='xset_val'))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_ntu120_xset_hrnet/j'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)
