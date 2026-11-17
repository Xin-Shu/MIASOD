# DINO-R50 (4-scale) for 512-px windows at native resolution, one object class.
# Fine-tuned from the COCO checkpoint for 12 epochs (AdamW, lr 1e-4, step at epoch 10, effective batch 8).
# Training: set data_root to a window set written by script/make_windows.py (COCO json per split), then
#   python mmdetection/tools/train.py script/detectors/dino/dino_r50_window512.py
_base_ = 'mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py'

size = 512
data_root = 'data/windows512/'
metainfo = dict(classes=('object', ))
load_from = ('https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/'
             'dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth')

model = dict(bbox_head=dict(num_classes=1), test_cfg=dict(max_per_img=300))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomResize', scale=(size, size), ratio_range=(0.7, 1.3), keep_ratio=True),
    dict(type='Pad', size=(size, size), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(size, size), keep_ratio=True),
    dict(type='Pad', size=(size, size), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=8, num_workers=4,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='train_coco.json', data_prefix=dict(img='train/'),
                 filter_cfg=dict(filter_empty_gt=False, min_size=32), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8, num_workers=4,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='val_coco.json', data_prefix=dict(img='val/'),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'val_coco.json')
test_evaluator = val_evaluator

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=4)
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[10], gamma=0.1)]
optim_wrapper = dict(optimizer=dict(lr=1e-4))
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
randomness = dict(seed=0)
