#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32

class PruneSubscriber:
    def __init__(self):
        self._prune_pos = None
        rospy.Subscriber('/prune_position', Float32, self._callback)

    def _callback(self, msg):
        self._prune_pos = msg.data

    def get_latest_position(self):
        return self._prune_pos