# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

from utils.ip_handler import get_ip, get_port
from utils.cam_mat_handler import get_x, get_z, get_theta, get_pitch

#-------------------------------------map----------------------------------------------
#座標変換用定数。predict_camera_locationで取得する(メインカメラについては、X,Y,THETA=0)
# X = 3.682768768475109
# Z = -2.6057855270816472
# THETA = 4.749273148314835
X = get_x()
Z = get_z()
THETA = get_theta()
PITCH = get_pitch()

#マップ生成サーバーのIP,ポート
CONNECT = True  #ソケット通信を行う場合はTrue、行わない場合はFalseにしてください。
SERVER_IP = get_ip()
SERVER_PORT = get_port()

CONTAIN_NO_BONE_DATA = False
PERSON_ID = '9'
#----------------------------------------------------------------------------------------

import argparse
import sys
import time
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets_rs import LoadStreams, LoadRealSense
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.point_handler import Point

#hrcode
import mediapipe as mp
import numpy as np

import pandas as pd


import pyrealsense2 as rs
import socket
import json

mp_drawing = mp.solutions.drawing_utils
mp_holistic =  mp.solutions.holistic

current_x = 0
current_z = 0
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def loop_handler():
    global sock
    while True:
        try:
            res = sock.recv(4096)
            res_data = res.decode('utf-8')
        except Exception as e:
            print(e)
            sock.close()
            break
        if res_data:
            if res_data == 'B':
                before_data = {
                    'bx' : X,
                    'bz' : Z,
                    'bt' : THETA
                }
                sock.send(json.dumps(before_data).encode('UTF-8'))
            elif res_data == 'C':
                current_data = {
                    'cx' : current_x,
                    'cz' : current_z
                }
                sock.send(json.dumps(current_data).encode('UTF-8'))
            elif ',' in res_data:
                print(res_data)

def get_3d_location(color_intr, depth_frame, pixel_x, pixel_y):
    if pixel_x==0 and pixel_y==0:
        return None
    distance = depth_frame.get_distance(pixel_x,pixel_y)
    if distance == 0:
        return None
    location_3d = rs.rs2_deproject_pixel_to_point(color_intr , [pixel_x,pixel_y], distance)
    return location_3d


def get_person_xy(person_list, results):
    x, y = 0, 0
    if person_list.shape[0] == 0:
        return x,y
    try:
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]
        x, y = (left_shoulder.x+right_shoulder.x)/2, (left_shoulder.y+right_shoulder.y)/2
    except:
        pass
    finally:
        if x == 0 and y == 0:
            pass
        else:
            data = np.where((x>=person_list[:,0]) & (y >= person_list[:,1]) & (x <= person_list[:,0]+person_list[:,2]) & (y <= person_list[:,1]+person_list[:,3]))
            if data[0].shape[0] == 0:
                x, y = 0, 0
        x = min(1, x)
        x = max(0, x)
        y = min(1, y)
        y = max(0, y)
        return x, y

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        tfl_int8=False,  # INT8 quantized TFLite model
        ):
    global current_x
    global current_z
    global sock
    if CONNECT:
        sock.connect((SERVER_IP, SERVER_PORT))
        thread = threading.Thread(target=loop_handler)
        thread.start()
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        #hrcode debug
        # print (model.names.index("book"))
        #hrcode debug
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadRealSense(source, img_size=imgsz, width=640, height=480, fps=30, stride=stride, auto=pt)
        # dataset = LoadRealSense(source, img_size=imgsz, width=1280, height=720, fps=30, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    wait_frame = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for path, depth, distance, depth_scale, img, im0s, vid_cap, color_intr in dataset:
            imt = im0s[0].copy()
            im_pose =cv2.cvtColor(imt, cv2.COLOR_BGR2RGB)
            results = holistic.process(im_pose)
            if wait_frame < 3 and view_img:
                wait_frame += 1
                continue
            else:
                wait_frame = 0
            if onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            t1 = time_sync()
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=augment, visualize=visualize)[0]
            elif onnx:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
            else:  # tensorflow model (tflite, pb, saved_model)
                imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
                if pb:
                    pred = frozen_func(x=tf.constant(imn)).numpy()
                elif saved_model:
                    pred = model(imn, training=False).numpy()
                elif tflite:
                    if tfl_int8:
                        scale, zero_point = input_details[0]['quantization']
                        imn = (imn / scale + zero_point).astype(np.uint8)
                    interpreter.set_tensor(input_details[0]['index'], imn)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    if tfl_int8:
                        scale, zero_point = output_details[0]['quantization']
                        pred = (pred.astype(np.float32) - zero_point) * scale
                pred[..., 0] *= imgsz[1]  # x
                pred[..., 1] *= imgsz[0]  # y
                pred[..., 2] *= imgsz[1]  # w
                pred[..., 3] *= imgsz[0]  # h
                pred = torch.tensor(pred)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()

            # Second-stage classifier (optional)
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            #[obj, x, y, z, w, h, conf, cx]
            box_list = np.empty((0,4), float)
            current_x = 0
            current_z = 0

            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                x=y=w=h=z=0
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                imt = im0.copy()
                hval = im0.shape[0]
                wval = im0.shape[1]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            #hrcode
                            rst = ('%g ' * len(line)).rstrip() % line + '\n'
                            rstlist = rst.split()
                            obj = rstlist[0]
                            x = float(xyxy[0]) / wval
                            y = float(xyxy[1]) / hval
                            w = float(xyxy[2] - xyxy[0]) / wval
                            h = float(xyxy[3] - xyxy[1]) / hval
                            if obj == PERSON_ID:
                                box_list = np.append(box_list, np.array([[x,y,w,h]]), axis=0)

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                #hrcode
                if view_img:
                    #最後に取得したresultsをim0にプロット
                    if results:
                        mp_drawing.draw_landmarks(im0,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                                )

                    location_3d = None
                    x0, y0= get_person_xy(box_list, results)
                    if not (x0 == 0 and y0 == 0):
                        location_3d = get_3d_location(color_intr, dataset.depth_frame, round((wval-1)*x0), round((hval-1)*y0))

                    if location_3d is not None:
                        current_x = location_3d[0]
                        current_z = location_3d[2]
                    
                    cv2.imshow('mpYolo', im0)
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--tfl-int8', action='store_true', help='INT8 quantized TFLite model')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
