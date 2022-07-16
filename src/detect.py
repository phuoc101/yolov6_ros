#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import os.path as osp
import sys
import time
from rostopic import get_topic_type

from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes

# add yolov5 submodule to path
ABS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = osp.join(ABS_DIR, 'yolov6')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer

@torch.no_grad()
class Yolov6Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.25)
        self.iou_thres = rospy.get_param("~iou_threshold", 0.45)
        self.agnostic_nms = rospy.get_param("~agnostic_nms", False)
        self.max_det = rospy.get_param("~maximum_detections", 1000)
        self.classes = rospy.get_param("~classes", None)
        self.half = rospy.get_param("~half", False)
        self.view_image = rospy.get_param("~view_image", True)
        self.yaml = rospy.get_param("~yaml", osp.join(ROOT, 'data/coco.yaml'))
        self.img_size = rospy.get_param("~img_size", 640)
        self.hide_conf = rospy.get_param("~hide_conf", False)
        self.hide_labels = rospy.get_param("~hide_labels", False)
        self.show_img = rospy.get_param("~show_img", True)
        # Initialize weights
        self.weights = osp.join(ROOT, rospy.get_param(
            "~weights", 'weights/yolov6n.pt'))
        # Initialize model
        self.device = rospy.get_param("~device", "gpu")
        self.inferer = Inferer(self.weights, self.device,
                               self.yaml, self.img_size, self.half)

        # Initialize subscriber to Image/CompressedImage topic
        self.img_type, self.img_topic, _ = get_topic_type(rospy.get_param(
            "~input_image_topic", "/image_raw"), blocking=True)
        self.compressed_input = self.img_type == "sensor_msgs/CompressedImage"
        1
        if self.compressed_input:
            self.img_sub = rospy.Subscriber(
                self.img_topic, CompressedImage, self.callback, queue_size=1
            )
        else:
            self.img_sub = rospy.Subscriber(
                self.img_topic, Image, self.img_cb, queue_size=1
            )
        self.bounding_boxes_pub = rospy.Publisher(
            rospy.get_param("~output_topic", 'yolo/detections'), BoundingBoxes, queue_size=10
        )
        # Initialize CV_Bridge
        self.bridge = CvBridge()

    def img_cb(self, msg):
        if self.compressed_input:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # cv2.imshow('original', frame)

        # result_frame, boxes, classes, proba = self.inferer.detect(frame, self.conf_thres, self.iou_thres, None,
        #                                                           self.agnostic_nms, self.max_det, False, False, False, False)
        det, img_src = self.inferer.detect(frame, self.conf_thres, self.iou_thres, None,
                                  self.agnostic_nms, self.max_det, False, False, False, False)
        img_ori = img_src

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = msg.header
        bounding_boxes.image_header = msg.header

        if len(det):
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)  # integer class
                class_name = self.inferer.class_names[class_num]
                label = None if self.hide_labels else (class_name if self.hide_conf else f'{class_name} {conf:.2f}')
                self.inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.inferer.generate_colors(class_num, True))
                img_src = np.asarray(img_ori)
                bounding_box = BoundingBox()
                bounding_box.Class = class_name
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])
                
                bounding_boxes.bounding_boxes.append(bounding_box)
        self.bounding_boxes_pub.publish(bounding_boxes)

        if self.show_img:
            cv2.imshow('yolov6', img_src)
            cv2.waitKey(1)

    def summary(self):
        return f"Yolov6 Detector summary:\n \
                Weights: {self.weights}\n \
                Confidence Threshold: {self.conf_thres}\n \
                IOU Threshold: {self.iou_thres}\n \
                Class-agnostic NMS: {self.agnostic_nms}\n \
                Maximal detections per image: {self.max_det}\n \
                Data yaml file: {self.yaml}"


def main():
    rospy.init_node("yolov6", anonymous=True, log_level = rospy.DEBUG)
    detector = Yolov6Detector()
    rospy.loginfo(detector.summary())
    rospy.spin()


if __name__ == "__main__":
    main()
