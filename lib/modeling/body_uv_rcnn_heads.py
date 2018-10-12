from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils


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
            15, 15, cfg.BODY_UV_RCNN.UP_SCALE)

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

        detectron_weight_mapping.update({
            'upsample_AnnIndex.upconv.weight': None,
            'upsample_AnnIndex.upconv.bias': None,
            'upsample_Index_UV.upconv.weight': None,
            'upsample_Index_UV.upconv.bias': None,
            'upsample_U.upconv.weight': None,
            'upsample_U.upconv.bias': None,
            'upsample_V.upconv.weight': None,
            'upsample_V.upconv.bias': None,
        })

        return detectron_weight_mapping, []

    def forward(self, x):
        Ann_Index = self.upsample_AnnIndex(self.deconv_AnnIndex(x))
        Index_UV = self.upsample_Index_UV(self.deconv_Index_UV(x))
        U_estimated = self.upsample_U(self.deconv_U(x))
        V_estimated = self.upsample_V(self.deconv_V(x))

        return U_estimated, V_estimated, Index_UV, Ann_Index


def body_uv_losses(U_estimated, V_estimated, Index_UV, Ann_Index,
                   body_uv_X_points,
                   body_uv_Y_points,
                   body_uv_I_points,
                   body_uv_Ind_points,
                   body_uv_U_points,
                   body_uv_V_points,
                   body_uv_point_weights,
                   body_uv_ann_labels,
                   body_uv_ann_weights):
    """Mask R-CNN body uv specific losses."""
    device_id = U_estimated.get_device()
    body_uv_X_points = Variable(torch.from_numpy(body_uv_X_points)).cuda(device_id)
    body_uv_Y_points = Variable(torch.from_numpy(body_uv_Y_points)).cuda(device_id)
    body_uv_I_points = Variable(torch.from_numpy(body_uv_I_points)).cuda(device_id)
    body_uv_Ind_points = Variable(torch.from_numpy(body_uv_Ind_points)).cuda(device_id)
    body_uv_U_points = Variable(torch.from_numpy(body_uv_U_points)).cuda(device_id)
    body_uv_V_points = Variable(torch.from_numpy(body_uv_V_points)).cuda(device_id)
    body_uv_point_weights = Variable(torch.from_numpy(body_uv_point_weights)).cuda(device_id)
    body_uv_ann_labels = Variable(torch.from_numpy(body_uv_ann_labels)).cuda(device_id)
    body_uv_ann_weights = Variable(torch.from_numpy(body_uv_ann_weights)).cuda(device_id)

    ## Reshape for GT blobs.
    X_points_reshaped = body_uv_X_points.view(-1, 1)
    Y_points_reshaped = body_uv_Y_points.view(-1, 1)
    I_points_reshaped = body_uv_I_points.view(-1, 1)
    Ind_points_reshaped = body_uv_Ind_points.view(-1, 1)
    ## Concat Ind,x,y to get Coordinates blob.
    Coordinates = torch.cat((Ind_points_reshaped, X_points_reshaped, Y_points_reshaped), dim=1)
    ##
    ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES
    ## U blob to
    ##
    U_points_reshaped = body_uv_U_points.view(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196)
    U_points_reshaped_transpose= torch.transpose(U_points_reshaped, 1, 2).contiguous()
    U_points = U_points_reshaped_transpose.view(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1)
    ## V blob
    ##
    V_points_reshaped = body_uv_V_points.view(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196)
    V_points_reshaped_transpose= torch.transpose(V_points_reshaped, 1, 2).contiguous()
    V_points = V_points_reshaped_transpose.view(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1)
    ###
    ## UV weights blob
    ##
    Uv_point_weights_reshaped = body_uv_point_weights.view(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196)
    Uv_point_weights_reshaped_transpose = torch.transpose(Uv_point_weights_reshaped, 1, 2).contiguous()
    Uv_point_weights = Uv_point_weights_reshaped_transpose.view(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1)

    #####################
    ###  Pool IUV for points via bilinear interpolation.
    Coordinates[:, 1:3] -= cfg.BODY_UV_RCNN.HEATMAP_SIZE/2.
    Coordinates[:, 1:3] *= 2./cfg.BODY_UV_RCNN.HEATMAP_SIZE
    grid = Coordinates[:, 1:3].view(-1, 14, 14, 2)
    interp_U = F.grid_sample(U_estimated, grid)
    interp_U = interp_U.permute(0,2,3,1).contiguous().view(-1, cfg.BODY_UV_RCNN.NUM_PATCHES+1)
    interp_V = F.grid_sample(V_estimated, grid)
    interp_V = interp_V.permute(0,2,3,1).contiguous().view(-1, cfg.BODY_UV_RCNN.NUM_PATCHES+1)
    interp_Index_UV = F.grid_sample(Index_UV, grid)
    interp_Index_UV = interp_Index_UV.permute(0,2,3,1).contiguous().view(-1, cfg.BODY_UV_RCNN.NUM_PATCHES+1)

    ## Reshape interpolated UV coordinates to apply the loss.

    interp_U_reshaped = interp_U.view(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1)
    interp_V_reshaped = interp_V.view(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1)
    ###

    ### Do the actual labels here !!!!
    body_uv_ann_labels_reshaped = body_uv_ann_labels.view(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE, cfg.BODY_UV_RCNN.HEATMAP_SIZE)
    body_uv_ann_weights_reshaped = body_uv_ann_weights.view(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE, cfg.BODY_UV_RCNN.HEATMAP_SIZE)
    ###
    I_points_reshaped_int = I_points_reshaped.to(torch.int64)
    ### Now add the actual losses
    ## The mask segmentation loss (dense)
    num_cls_Index = Ann_Index.size(1)
    Ann_Index_reshaped = Ann_Index.permute(0,2,3,1).contiguous().view(-1, num_cls_Index)
    body_uv_ann_labels_reshaped_int = body_uv_ann_labels_reshaped.to(torch.int64)
    loss_seg_AnnIndex = F.cross_entropy(Ann_Index_reshaped, body_uv_ann_labels_reshaped_int.view(-1))
    loss_seg_AnnIndex *= cfg.BODY_UV_RCNN.INDEX_WEIGHTS

    ## Point Patch Index Loss.
    loss_IndexUVPoints = F.cross_entropy(interp_Index_UV, I_points_reshaped_int.squeeze())
    loss_IndexUVPoints *= cfg.BODY_UV_RCNN.PART_WEIGHTS
    ## U and V point losses.
    loss_Upoints = net_utils.smooth_l1_loss(interp_U_reshaped, U_points, Uv_point_weights, Uv_point_weights)
    loss_Upoints *= cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS

    loss_Vpoints = net_utils.smooth_l1_loss(interp_V_reshaped, V_points, Uv_point_weights, Uv_point_weights)
    loss_Vpoints *= cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS

    return loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints


# ---------------------------------------------------------------------------- #
# Body uv heads
# ---------------------------------------------------------------------------- #

class roi_body_uv_head_v1convX(nn.Module):
    """Mask R-CNN Body uv head. v1convX design: X * (conv)."""
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
            detectron_weight_mapping['conv_fcn.%d.weight' % (2*i)] = 'body_conv_fcn%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn.%d.bias' % (2*i)] = 'body_conv_fcn%d_b' % (i+1)

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
