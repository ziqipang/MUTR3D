_base_ = [
    '../../_base_/datasets/nus-3d.py',
    '../../_base_/default_runtime.py'
]
workflow = [('train', 1)]
plugin = True
plugin_dir = 'plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='MUTRPETRCamTracker',
    use_grid_mask=True,  # use grid mask
    num_classes=7,
    num_query=300,
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=100,
        num_classes=7),
    fix_feats=False, # set fix feats to true can fix the backbone
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.0, random_drop=0.0), # hyper-param for query dropping mentioned in MOTR
    mem_cfg=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    radar_encoder=dict( # not used in this project. Provide a framework for fusing radar features
        type='RadarPointEncoderXY',
        in_channels=13,
        out_channels=[32],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),),
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        # start_level=1,
        # add_extra_convs='on_output',
        # extra_convs_on_inputs=False,
        num_outs=2,),
        # norm_cfg=dict(type='BN2d'),
        # relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='PETRCamTrackingHead',
        num_classes=7,
        in_channels=256,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=7), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='GIoU3DCost', weight=0.0),
            pc_range=point_cloud_range)),
    test_cfg=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range[:2],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=voxel_size,
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2))


# config for radar data. not used in this repo
# x y z rcs vx vy vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
dataset_type = 'NuScenesTrackDataset'
data_root = 'data/nuscenes/'
ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
}

file_client_args = dict(backend='disk')

train_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=1,
    #     use_dim=radar_use_dims,
    #     max_num=100, ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    #dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'radar'])
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t'])
]
test_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=1,
    #     use_dim=radar_use_dims,
    #     max_num=100, ),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['points', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'radar'])
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            num_frames_per_sample=2,  # number of frames for each clip in training. If you have more memory, I suggested you to use more.
            data_root=data_root,
            ann_file=data_root + 'track_mutr_infos_train.pkl',
            pipeline_single=train_pipeline,
            pipeline_post=train_pipeline_post,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR'),
    # ),
    val=dict(type=dataset_type, pipeline_single=test_pipeline, pipeline_post=test_pipeline_post, classes=class_names, modality=input_modality,
             ann_file=data_root + 'track_mutr_infos_val.pkl',
             num_frames_per_sample=1,), # when inference, set bs=1
    test=dict(type=dataset_type, pipeline_single=test_pipeline,
              pipeline_post=test_pipeline_post,
              classes=class_names, modality=input_modality,
              ann_file=data_root + 'track_mutr_infos_val.pkl',
              num_frames_per_sample=1,)) # when inference, set bs=1

optimizer = dict(
    type='AdamW',
    #type='SGD',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=105, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 12 # I suggest you to train longer. Like 48, 72 epochs, and change lr_config accrodingly
evaluation = dict(interval=12)

runner = dict(type='EpochBasedRunner', max_epochs=12)

find_unused_parameters = True
load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth' # path to pretrained model.

# fp16 = dict(loss_scale='dynamic')
