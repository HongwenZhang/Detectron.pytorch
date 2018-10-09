from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn


# ---------------------------------------------------------------------------- #
# Body UV R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class body_uv_outputs(nn.Module):
    """Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    def __init__(self, dim_in):
        super(body_uv_outputs, self).__init__()

        # Apply ConvTranspose to the feature representation; results in 2x # upsampling
        self.deconv_AnnIndex = nn.ConvTranspose2d(
            dim_in, 15, cfg.BODY_UV_RCNN.DECONV_KERNEL,
            2, padding=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1))

        self.deconv_Index_UV = nn.ConvTranspose2d(
            dim_in, cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.DECONV_KERNEL,
            2, padding=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1))

        self.deconv_U = nn.ConvTranspose2d(
            dim_in, cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.DECONV_KERNEL,
            2, padding=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1))

        self.deconv_V = nn.ConvTranspose2d(
            dim_in, cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.DECONV_KERNEL,
            2, padding=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1))
        ###
        self.upsample_AnnIndex = mynn.BilinearInterpolation2d(
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.UP_SCALE)

        self.upsample_Index_UV = mynn.BilinearInterpolation2d(
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.UP_SCALE)

        self.upsample_U = mynn.BilinearInterpolation2d(
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.UP_SCALE)

        self.upsample_V = mynn.BilinearInterpolation2d(
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1, cfg.BODY_UV_RCNN.UP_SCALE)

        self._init_weights()

    def _init_weights(self):

        if cfg.BODY_UV_RCNN.CONV_INIT == 'GaussianFill':
            init.normal_(self.deconv_AnnIndex.weight, std=0.001)
            init.normal_(self.deconv_Index_UV.weight, std=0.001)
            init.normal_(self.deconv_U.weight, std=0.001)
            init.normal_(self.deconv_V.weight, std=0.001)
        elif cfg.BODY_UV_RCNN.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.deconv_AnnIndex.weight)
            mynn.init.MSRAFill(self.deconv_Index_UV.weight)
            mynn.init.MSRAFill(self.deconv_U.weight)
            mynn.init.MSRAFill(self.deconv_V.weight)
        else:
            raise ValueError(cfg.KRCNN.CONV_INIT)

        init.constant_(self.deconv_AnnIndex.bias, 0)
        init.constant_(self.deconv_Index_UV.bias, 0)
        init.constant_(self.deconv_U.bias, 0)
        init.constant_(self.deconv_V.bias, 0)


    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}

        detectron_weight_mapping.update({
            'deconv_AnnIndex.weight': 'AnnIndex_lowres_w',
            'deconv_AnnIndex.bias': 'AnnIndex_lowres_b',
            'deconv_Index_UV.weight': 'Index_UV_lowres_w',
            'deconv_Index_UV.bias': 'Index_UV_lowres_b',
            'deconv_U.weight': 'U_lowres_w',
            'deconv_U.bias': 'U_lowres_b',
            'deconv_V.weight': 'V_lowres_w',
            'deconv_V.bias': 'V_lowres_b',
        })

        return detectron_weight_mapping, []

    def forward(self, x):
        Ann_Index = self.upsample_AnnIndex(self.deconv_AnnIndex(x))
        Index = self.upsample_Index_UV(self.deconv_Index_UV(x))
        U = self.upsample_U(self.deconv_U(x))
        V = self.upsample_V(self.deconv_V(x))

        return U, V, Index, Ann_Index


def body_uv_losses(kps_pred, keypoint_locations_int32, keypoint_weights,
                    keypoint_loss_normalizer=None):
    """Mask R-CNN keypoint specific losses."""
    device_id = kps_pred.get_device()
    kps_target = Variable(torch.from_numpy(
        keypoint_locations_int32.astype('int64'))).cuda(device_id)
    keypoint_weights = Variable(torch.from_numpy(keypoint_weights)).cuda(device_id)
    # Softmax across **space** (woahh....space!)
    # Note: this is not what is commonly called "spatial softmax"
    # (i.e., softmax applied along the channel dimension at each spatial
    # location); This is softmax applied over a set of spatial locations (i.e.,
    # each spatial location is a "class").
    loss = F.cross_entropy(
        kps_pred.view(-1, cfg.KRCNN.HEATMAP_SIZE**2), kps_target, reduce=False)
    loss = torch.sum(loss * keypoint_weights) / torch.sum(keypoint_weights)
    loss *= cfg.KRCNN.LOSS_WEIGHT

    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        loss *= keypoint_loss_normalizer.item() # np.float32 to float
    return loss


# ---------------------------------------------------------------------------- #
# Keypoint heads
# ---------------------------------------------------------------------------- #

class roi_body_uv_head_v1convX(nn.Module):
    """Mask R-CNN keypoint head. v1convX design: X * (conv)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super(roi_body_uv_head_v1convX, self).__init__()

        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
        kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL

        pad_size = kernel_size // 2
        module_list = []
        for _ in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.conv_fcn = nn.Sequential(*module_list)
        self.dim_out = hidden_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.BODY_UV_RCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.BODY_UV_RCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.KRCNN.CONV_INIT: {}'.format(cfg.KRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
            detectron_weight_mapping['conv_fcn.%d.weight' % (2*i)] = 'conv_fcn%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn.%d.bias' % (2*i)] = 'conv_fcn%d_b' % (i+1)

        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='body_uv_rois',
            method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return x
