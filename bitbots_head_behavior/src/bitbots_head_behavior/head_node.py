#!/usr/bin/env python
"""
This is the ROS-Node which contains the head behavior, starts the appropriate DSD, initializes the HeadBlackboard
and subscribes to head_behavior specific ROS-Topics.
"""
import os

import rospy

from bitbots_blackboard.blackboard import HeadBlackboard
from dynamic_stack_decider.dsd import DSD

from humanoid_league_msgs.msg import HeadMode as HeadModeMsg, BallRelative
from bitbots_msgs.msg import JointCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def run(dsd):
    """
    Main run-loop

    :returns: Never
    """
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        dsd.update()
        rate.sleep()


def init():
    """
    Initialize new components needed for head_behavior:
    blackboard, dsd, rostopic subscriber
    """
    rospy.init_node('head_behavior')
    blackboard = HeadBlackboard()

    rospy.Subscriber('/head_mode', HeadModeMsg, blackboard.head_capsule.head_mode_callback, queue_size=1)
    rospy.Subscriber("/ball_relative", BallRelative, blackboard.world_model.ball_callback)
    rospy.Subscriber('/joint_states', JointState, blackboard.head_capsule.joint_state_callback)
    blackboard.head_capsule.position_publisher = rospy.Publisher("/head_motor_goals", JointCommand, queue_size=10)
    blackboard.head_capsule.visual_compass_record_trigger = rospy.Publisher( blackboard.config['visual_compass_trigger_topic'], Header, queue_size=5)

    dirname = os.path.dirname(os.path.realpath(__file__))

    dsd = DSD(blackboard, '/debug/dsd/head_behavior')
    dsd.register_actions(os.path.join(dirname, 'actions'))
    dsd.register_decisions(os.path.join(dirname, 'decisions'))
    dsd.load_behavior(os.path.join(dirname, 'head_behavior.dsd'))

    rospy.logdebug("Head Behavior completely loaded")
    return dsd


if __name__ == '__main__':
    run(init())
