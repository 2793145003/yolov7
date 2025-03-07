import os
import os.path as osp
import tempfile
import json
from argparse import ArgumentParser

import mmcv
import cv2
import numpy as np

from mmtrack.apis import init_model
from mmtrack.core import results2outs
from mmdet.datasets.pipelines import Compose

import torch
from mmcv.parallel import collate, scatter

from yolov7_model import YoloV7
from yolo_sort import YOLOSORT


def change_video(input_path, output_name, fps, size):
    print(f'changing video {input_path} fps {fps} size {size}')
    ffmpeg_command = (f'ffmpeg -i {input_path} -s {size[0]}x{size[1]} -b 300k -r {fps} -y {output_name}')
    os.system(ffmpeg_command)
    return output_name

def main():
    parser = ArgumentParser()
    parser.add_argument('--fps', help='fps', default=10)
    parser.add_argument('--width', help='width', default=960)
    parser.add_argument('--height', help='height', default=540)
    parser.add_argument('--input_path', help='input path', default='/input/VID_20220727_175808.mp4')
    parser.add_argument('--output_path', help='output path', default='/input')
    parser.add_argument('--counter_path', help='counter json path, in range [0-1]', default='/input/counter.json')
    parser.add_argument('--batch_size', help='detector batch size', default=32)
    args = parser.parse_args()

    with open(f"{args.output_path}/progress.txt", "w") as f:
        f.write("0\n")

    config = 'configs/yolov7.py'
    input_path = args.input_path
    fps = int(args.fps)
    batch_size = int(args.batch_size)
    width = int(args.width)
    height = int(args.height)
    size = (width, height)
    score_thr = 0.
    device = 'cuda:0'
    show = False
    backend = 'cv2'
    checkpoint = None

    input_path = change_video(input_path, "changed_video.mp4", fps, size)
    with open(f"{args.output_path}/progress.txt", "a") as f:
        f.write("10\n")

    cnt = None
    try:
        with open(args.counter_path) as f:
            counter_json = json.load(f)
            xs = np.array(counter_json['xs']).reshape([-1, 1])
            ys = np.array(counter_json['ys']).reshape([-1, 1])
            cnt = np.concatenate((xs, ys), axis=1) * size
            cnt = cnt.reshape([-1, 1, 2]).astype(np.int64)
    except:
        print("can't find counter file.")

    imgs = mmcv.VideoReader(input_path)
    IN_VIDEO = True
    OUT_VIDEO = True
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name
    _out = 'test.mp4'.rsplit(os.sep, 1)
    if len(_out) > 1:
        os.makedirs(_out[0], exist_ok=True)
    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    all_ids = dict()
    id_count = []
    result_frame_id = 0
    batch_img = []
    for frame_id, img in enumerate(imgs):
        prog_bar.update()
        if frame_id % batch_size == 0:
            batch = []
            batch_img = []

        if isinstance(img, str):
            img = osp.join(input_path, img)
        # ------------------------------------
        # result = inference_mot(model, img, frame_id=frame_id)
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img, img_info=dict(frame_id=frame_id), img_prefix=None)
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename=img, frame_id=frame_id), img_prefix=None)
        # build the data pipeline
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        batch.append(data)
        batch_img.append(img)
        if len(batch) < batch_size:
            continue

        with open(f"{args.output_path}/progress.txt", "a") as f:
            f.write(f"{10+int(frame_id/len(imgs)*80)}\n")

        data = collate(batch, samples_per_gpu=batch_size)
        data = scatter(data, [device])[0]

        # forward the model
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
        # ------------------------------------

        for result, img in zip(results, batch_img):
            out_file = osp.join(out_path, f'{result_frame_id:06d}.jpg')

            ## 2.提取结果并计数
            track_bboxes = result.get('track_bboxes', None)
            track_masks = result.get('track_masks', None)
            outs_track = results2outs(
                    bbox_results=track_bboxes,
                    mask_results=track_masks,
                    mask_shape=img.shape[:2])
            show_ids=[]
            for i in range(len(outs_track.get('labels', None))):
                label =  outs_track.get('labels', None)[i]
                bbox =  outs_track.get('bboxes', None)[i]
                id =  outs_track.get('ids', None)[i]
                x1, y1, x2, y2, conf = bbox
                if conf < 0.9:
                    continue
                # 保存小图
                # crop = img[int(y1):int(y2), int(x1):int(x2)]
                # if not os.path.exists(f'crop/{args.input_name}/'):
                #     os.system(f'mkdir crop/{args.input_name}')
                # if not os.path.exists(f'crop/{args.input_name}/id_{id}/'):
                #     os.system(f'mkdir crop/{args.input_name}/id_{id}')
                # cv2.imwrite(f'crop/{args.input_name}/id_{id}/frame_{frame_id}.jpg', crop)
                
                # 如果下边中点在框外面
                if cnt is not None and cv2.pointPolygonTest(cnt, ((x1+x2)/2, y2), False) < 0:
                    continue

                if id not in all_ids:
                    all_ids[id] = 0
                all_ids[id] += 1
                if all_ids[id] > fps:
                    show_ids.append(id)
                    if id not in id_count:
                        id_count.append(id)
            show_ids = sorted(show_ids)
            for i in range(len(show_ids)):
                img = cv2.putText(img, f"{show_ids[i]}", (50*i, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 0, 255), 2)

            img = cv2.putText(img, f"count: {len(id_count)}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 2)
            img = cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
            img = cv2.putText(img, f"{0}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            model.show_result(
                img,
                result,
                score_thr=score_thr,
                show=show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=backend)
            cv2.imwrite(out_file, img)
            result_frame_id += 1
        # break

    ## 3.输出结果
    count = 0
    for key in all_ids:
        if all_ids[key] > fps: # 只计算这些帧以上的
            count += 1
            # print(key)
    with open(f"{args.output_path}/count.txt", "w") as f:
        f.write(f"{count}")
    
    with open(f"{args.output_path}/progress.txt", "a") as f:
        f.write(f"{90}\n")

    mmcv.frames2video(out_path, f'{args.output_path}/result.mp4', fps=fps, fourcc='mp4v')
    out_dir.cleanup()
    with open(f"{args.output_path}/progress.txt", "a") as f:
        f.write(f"{100}")


if __name__ == '__main__':
    main()

