#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32

class GripperPoseSubscriber:
    def __init__(self):
        self._gripper_pos = None
        rospy.Subscriber('/gripper_position', Float32, self._callback)

    def _callback(self, msg):
        self._gripper_pos = msg.data

    def get_latest_position(self):
        return self._gripper_pos