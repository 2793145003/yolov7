from models.experimental import attempt_load
from utils.general import non_max_suppression, clip_coords
from utils.torch_utils import select_device

from mmdet.models.builder import DETECTORS

@DETECTORS.register_module()
class YoloV7():
    def __init__(self, weight, 
                 img_size=1280, 
                 conf_thres=0.25, 
                 iou_thres=0.45, 
                 classes=0, **args):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes

        device = select_device('0')
        self.model = attempt_load(weight, map_location=device)
        # self.model = TracedModel(self.model, device, img_size)

    def extract_feat(self, img):
        # print(img.shape)
        preds = self.model(img, augment=False)[0]
        results = []
        for i in range(len(preds)):
            pred = preds[i:i+1, :, :]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)[0]
            pred[:, :4] = scale_coords(img.shape[-2:], pred[:, :4], [540, 960]).round()
            results.append(pred)
        return results

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # coords[:, [0, 2]] -= pad[0]  # x padding
    # coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

