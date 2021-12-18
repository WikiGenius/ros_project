#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from math import cos, sin, atan2


class DetectTapes:

    def __init__(self):
        print('X')
        # CV bridge object
        self.cvBridge = CvBridge()
        rospy.Subscriber('/image_publisher_1639792718765387742/image_raw/compressed',
                         CompressedImage, self._img_callback)
        # self.pub_img_proj = rospy.Publisher(
        #     '/camera/image_output/compressed', CompressedImage, queue_size=1)

    def _img_callback(self, msg):
        # converts compressed image to opencv image
        np_img_input = np.frombuffer(msg.data, np.uint8)
        cv_img_input = cv2.imdecode(np_img_input, cv2.IMREAD_COLOR)
        print(cv_img_input.shape)
        print(cv_img_input)
        print(msg.header.frame_id)
        # adding Gaussian blur to the image of original
        cv_img_input = cv2.GaussianBlur(cv_img_input, (3, 3), 0)
        rospy.loginfo(cv_img_input.shape)

        # Display the resulting frame

        # Display the resulting frame
        cv2.imshow('frame', cv_img_input)

        # Press Q on keyboard to stop recording

        if cv2.waitKey(1) & 0xFF == ord('q'):

            exit()


if __name__ == '__main__':
    # Node initialization
    rospy.init_node('test_subscribe_video')
    detect_tapes = DetectTapes()

    # Looping
    rospy.spin()
