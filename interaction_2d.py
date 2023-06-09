# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

#hrcode
import mediapipe as mp
import numpy as np
import os

import csv
import pickle
import pandas as pd

import math

# with open('pkl/action_2d.pkl', 'rb') as f0:
#     model_action = pickle.load(f0)
# with open('pkl/cellphone_2d.pkl', 'rb') as f1:
#    model_cellphone = pickle.load(f1)
with open('pkl/bottle_2d.pkl', 'rb') as f2:
    model_bottle = pickle.load(f2)
# with open('pkl/book_2d.pkl', 'rb') as f3:
#     model_book = pickle.load(f3)
# with open('pkl/keyboard_2d.pkl', 'rb') as f4:
#     model_keyboard = pickle.load(f4)

def get_distance(x1, y1, x2, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d

def predict_interaction(x_var, y_var, w_var, h_var, model_var, results, wval, hval, coords_s_pre, distance_s, interaction_class_out, interaction_prob_out):
    interaction_class = "None"
    interaction_prob = (0,0,0)

    pose = results.pose_landmarks.landmark
    coords_s = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    coords_obj=list(np.array([x_var, y_var]).flatten())
    shape_obj=list(np.array([w_var, h_var]).flatten())
    
    for rpi in range (4,132):
        rpj=0 if rpi%4==0 else 1 if rpi%4==1 else 2 if rpi%4==2 else 3
        if rpj!=3:
            coords_s[rpi]=coords_s[rpj]-coords_s[rpi]
    distance_so= list()
    for landmark in pose:
        pose_2d=np.array([landmark.x,landmark.y])
        object_2d=np.array([float(x_var+w_var/2), float(y_var+h_var/2)])
        distance_so.append(np.linalg.norm(pose_2d-object_2d))
    nosex = round(wval*pose[mp_holistic.PoseLandmark.NOSE.value].x)
    nosey = round(hval*pose[mp_holistic.PoseLandmark.NOSE.value].y)
    coords_human = [nosex, nosey]

    # distance from previous frame pose landmark
    for i4 in range (len(coords_s)):
        distance_s[i4]=coords_s[i4]-coords_s_pre[i4]
    coords_s[0]=coords_s[1]=coords_s[2]=0
    row = coords_s+coords_human+coords_obj+shape_obj+distance_so+distance_s
    X = pd.DataFrame([row])

    interaction_class= model_var.predict(X)[0]
    interaction_prob = model_var.predict_proba(X)[0]
    if(interaction_class!="None"):
        interaction_class_out.append(model_var.predict(X)[0])
        interaction_prob_out.append(round(interaction_prob[np.argmax(interaction_prob)],2))

def predict_action(model_var, results, wval, hval, coords_s_pre, distance_s, action_class, action_prob):

    pose = results.pose_landmarks.landmark
    coords_s = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    for rpi in range (4,132):
        rpj=0 if rpi%4==0 else 1 if rpi%4==1 else 2 if rpi%4==2 else 3
        if rpj!=3:
            coords_s[rpi]=coords_s[rpj]-coords_s[rpi]

    # distance from previous frame pose landmark
    for i4 in range (len(coords_s)):
        distance_s[i4]=coords_s[i4]-coords_s_pre[i4]
    coords_s[0]=coords_s[1]=coords_s[2]=0

    nosex = round(wval*pose[mp_holistic.PoseLandmark.NOSE.value].x)
    nosey = round(hval*pose[mp_holistic.PoseLandmark.NOSE.value].y)
    coords_human = [nosex, nosey]

    row = coords_s + coords_human + distance_s
    X = pd.DataFrame([row])

    action_class = model_var.predict(X)[0]
    action_prob = model_var.predict_proba(X)[0]
    action_prob = round(action_prob[np.argmax(action_prob)],2)
    return action_class, action_prob

mp_drawing = mp.solutions.drawing_utils
mp_holistic =  mp.solutions.holistic
#hrcode

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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    distance_s=[0 for i in range(132)]
    coords_s_pre=[0 for i in range(132)]
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for path, img, im0s, vid_cap in dataset:
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

            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                x=y=w=h=0
                x2=y2=w2=h2=0
                x3=y3=w3=h3=0
                x4=y4=w4=h4=0
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
                            hval = im0.shape[0]
                            wval = im0.shape[1]
                            if obj == '0': 
                                x0 = float(xyxy[0]) / wval
                                y0 = float(xyxy[1]) / hval
                                w0 = float(xyxy[2] - xyxy[0]) / wval
                                h0 = float(xyxy[3] - xyxy[1]) / hval
                            if obj == '67': #cell phone      
                                x = float(xyxy[0]) / wval
                                y = float(xyxy[1]) / hval
                                w = float(xyxy[2] - xyxy[0]) / wval
                                h = float(xyxy[3] - xyxy[1]) / hval
                            if obj == '73': #book
                                x2 = float(xyxy[0]) / wval
                                y2 = float(xyxy[1]) / hval
                                w2 = float(xyxy[2] - xyxy[0]) / wval
                                h2 = float(xyxy[3] - xyxy[1]) / hval
                            if obj == '66': #keyboard
                                x3 = float(xyxy[0]) / wval
                                y3 = float(xyxy[1]) / hval
                                w3 = float(xyxy[2] - xyxy[0]) / wval
                                h3 = float(xyxy[3] - xyxy[1]) / hval
                            if obj == '1': #cup
                                x4 = float(xyxy[0]) / wval
                                y4 = float(xyxy[1]) / hval
                                w4 = float(xyxy[2] - xyxy[0]) / wval
                                h4 = float(xyxy[3] - xyxy[1]) / hval
                            #hrcode

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
                    action_class, action_prob, interaction_class_out, interaction_prob_out, body_language_class_all, body_language_prob_all= "None",  0.0, [], [], "None", 0.0
                    hval = im0.shape[0]
                    wval = im0.shape[1]
                    im0 =cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    results = holistic.process(im0)
                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(im0,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                            )
                    try:
                        # action_class, action_prob = predict_action(model_action,results, wval,hval,coords_s_pre, distance_s, action_class, action_prob)
                        # predict_interaction(x,y,w,h,model_cellphone,results, wval,hval,coords_s_pre, distance_s, interaction_class_out, interaction_prob_out)
                        # predict_interaction(x2,y2,w2,h2,model_book,results, wval,hval,coords_s_pre, distance_s, interaction_class_out, interaction_prob_out)
                        # predict_interaction(x3,y3,w3,h3,model_keyboard,results, wval,hval,coords_s_pre, distance_s, interaction_class_out, interaction_prob_out)
                        predict_interaction(x4,y4,w4,h4,model_bottle,results, wval,hval,coords_s_pre, distance_s, interaction_class_out, interaction_prob_out)
                        if(interaction_class_out):
                            body_language_prob_all=max(interaction_prob_out)
                            body_language_class_all=interaction_class_out[interaction_prob_out.index(max(interaction_prob_out))]
                    except:
                        pass


                    im0 = cv2.copyMakeBorder(im0, 260, 0, 0, 0, cv2.BORDER_CONSTANT, value=[245, 117, 16])

                    # Display Class
                    cv2.putText(im0, 'Interaction'
                                , (155,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.putText(im0, body_language_class_all
                                , (150,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(im0, 'Probability'
                                , (15,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.putText(im0, str(body_language_prob_all)
                                , (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)

                    cv2.putText(im0, 'Action'
                                , (155,22+120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.putText(im0, action_class
                                , (150,90+120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(im0, 'Probability'
                                , (15,22+120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.putText(im0, str(action_prob)
                                , (10,90+120), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)

                    cv2.imshow('mpYolo', im0)
                        # cv2.waitKey(1)  # 1 millisecond
                #hrcode

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
