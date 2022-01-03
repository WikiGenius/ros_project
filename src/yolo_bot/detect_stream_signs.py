#!/usr/bin/env python3

"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

from cv_bridge import CvBridge
from std_msgs.msg import Int16
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
import numpy as np
import rospy
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, load_stream_images
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class DetectSigns:

    def __init__(self):
        # CV bridge object
        self.cvBridge = CvBridge()
        # self.topic_in_image = '/image_publisher_1639846134116251671/image_raw/compressed'
        self.topic_in_image = '/realsense/color/image_raw/compressed'
        self.topic_out_image = '/camera/image_output/compressed'
        self.topic_out_planner = '/class_name'
        self.debug = False
        self.total_time_process = time_sync()

        opt = self.parse_opt()
        self.define_variables(**vars(opt))

        rospy.Subscriber(self.topic_in_image, CompressedImage, self._img_callback)
        self.pub_img_proj = rospy.Publisher(self.topic_out_image, CompressedImage, queue_size=1)
        self.pub_sign = rospy.Publisher(self.topic_out_planner, Int16, queue_size=1)

    def _img_callback(self, msg):
        self.total_time_process = time_sync() - self.total_time_process
        if self.debug:
            print("image subscribed")
            print(f"total_time_process : {round(self.total_time_process)}s --- {1/self.total_time_process:.2f} FPS") 
        # converts compressed image to opencv image
        np_img_input = np.frombuffer(msg.data, np.uint8)
        cv_img_input = cv2.imdecode(np_img_input, cv2.IMREAD_COLOR)
            
        # ==========================================================================
        cv_img_output = self.run(cv_img_input)
        # Yolo process on image stream from gazebo

        # ==========================================================================
        # Publish the detected image
        self.pub_img_proj.publish(
            self.cvBridge.cv2_to_compressed_imgmsg(cv_img_output, "jpg"))

    def define_variables(self, weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                         source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                         imgsz=(640, 640),  # inference size (height, width)
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
                         project=ROOT / 'runs/detect',  # save results to project/name
                         name='exp',  # save results to project/name
                         exist_ok=False,  # existing project/name ok, do not increment
                         line_thickness=3,  # bounding box thickness (pixels)
                         hide_labels=False,  # hide labels
                         hide_conf=False,  # hide confidences
                         half=False,  # use FP16 half-precision inference
                         dnn=False,  # use OpenCV DNN for ONNX inference
                         ):
        self.source = str(source)
        self.save_img = not nosave
        self.is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.txt') or (self.is_url and not self.is_file)
        self.weights = weights
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, self.jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.names[1] = 'Speed limit (16km/h)'
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        self.dt, self.seen = [0.0, 0.0, 0.0], 0
        self.bs = 1
        self.half = half
        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        # Half
        # half precision only supported by PyTorch on CUDA
        self.half &= (self.pt or self.jit or engine) and self.device.type != 'cpu'
        if self.pt or self.jit:
            self.model.model.half() if half else self.model.model.float()

        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.view_img = view_img
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.visualize = visualize
        self.update = update
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        # self.out = cv2.VideoWriter('recorded.avi', -1, 20.0, (640,480))
        self.out = cv2.VideoWriter('recorded.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
        # ==============================================================

    @torch.no_grad()
    def run(self, im0s):

        # ==========================subscribe====================================
        # Dataloader
        
        im = load_stream_images(im0s)
        s=''
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        self.dt[1] += t3 - t2
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        
        for _, det in enumerate(pred):  # per image
            s += f'{im.shape[2:]} '   # print string
            self.seen += 1
            im0 = im0s.copy()

            save_path = str(self.save_dir)  # im.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    # Publish the class message
                    self.pub_sign.publish(int(c.item()))
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (
                            self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c], BGR=True)
            # Print time (inference-only)
            # LOGGER.info(f'{s} detection time. ({t3 - t2:.3f}s)')
            print(f'{s} detection time per frame. ({(t3 - t2)*1000:.3f}ms)  -- {int(1/(t3 - t2))} FPS')
            # Stream results
            im0 = annotator.result()
            if self.view_img:
                # Display the resulting frame
                cv2.imshow('detection', im0)

                # Press Q on keyboard to stop recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit()
             
            # Save results (image with detections)
            if self.save_img:
                # cv2.imwrite(save_path+'/recorded.mp4', im0)
                self.out.write(im0)

        return im0

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                            type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(FILE.stem, opt)
        return opt

    def run_node(self):

        try:
            # Looping
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS detect-signs module")
        cv2.destroyAllWindows()

        # Print results
        t = tuple(x / self.seen * 1E3 for x in self.dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    # Node initialization
    rospy.init_node('detect_signs')

    detect_signs = DetectSigns()

    # opt = detect_signs.parse_opt()
    check_requirements(exclude=('tensorboard', 'thop'))
    # detect_signs.define_variables(**vars(opt))

    detect_signs.run_node()
