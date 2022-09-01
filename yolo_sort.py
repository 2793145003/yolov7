# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models import build_detector

from mmtrack.core import outs2results
from mmtrack.models.builder import MODELS, build_motion, build_reid, build_tracker
from mmtrack.models.mot.base import BaseMultiObjectTracker

import torch

@MODELS.register_module()
class YOLOSORT(BaseMultiObjectTracker):
    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.time = dict(detector=[], tracker=[])
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            if detector:
                detector_pretrain = pretrains.get('detector', None)
                if detector_pretrain:
                    detector.init_cfg = dict(
                        type='Pretrained', checkpoint=detector_pretrain)
                else:
                    detector.init_cfg = None
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None

        if detector is not None:
            self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models firstly, then \
                inference with SORT/DeepSORT.')

    def simple_test(self,
                    imgs,
                    img_metases,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        # [xyxy, conf, cls]
        # start = time.time()
        xs = self.detector.extract_feat(imgs)
        # self.time['detector'].append(time.time()-start)
        results = []
        for i in range(len(xs)):
            x = xs[i]
            img = torch.unsqueeze(imgs[i], 0)
            img_metas = [img_metases[i]]
            frame_id = img_metas[0].get('frame_id', -1)
            if frame_id == 0:
                self.tracker.reset()

            det_bboxes = x[:, :5]
            det_labels = x[:, 5:]

            # start = time.time()
            track_bboxes, track_labels, track_ids = self.tracker.track(
                img=img,
                img_metas=img_metas,
                model=self,
                feats=x,
                bboxes=det_bboxes,
                labels=det_labels,
                frame_id=frame_id,
                rescale=rescale,
                **kwargs)
            # self.time['tracker'].append(time.time()-start)
            track_results = outs2results(
                bboxes=track_bboxes,
                labels=track_labels.squeeze(1),
                ids=track_ids,
                num_classes=1)
            det_results = outs2results(
                bboxes=det_bboxes, labels=det_labels.squeeze(1), num_classes=1)

        # if len(self.time["detector"]) % 100 == 0:
        #     print(f'\n-> detector_min: {min(self.time["detector"])}, detector_max: {max(self.time["detector"])}, detector_mean: {statistics.mean(self.time["detector"])}')
        #     print(f'-> tracker_min: {min(self.time["tracker"])}, tracker_max: {max(self.time["tracker"])}, tracker_mean: {statistics.mean(self.time["tracker"])}')

        results.append(dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results']))
        return results

