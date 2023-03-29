#!/usr/bin/env python3
"""
This is the ROS-Node which contains the head behavior, starts the appropriate DSD, initializes the HeadBlackboard
and subscribes to head_behavior specific ROS-Topics.
"""
import os

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from bitbots_blackboard.blackboard import HeadBlackboard
from dynamic_stack_decider.dsd import DSD

from humanoid_league_msgs.msg import HeadMode as HeadModeMsg
from bitbots_msgs.msg import JointCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from ament_index_python import get_package_share_directory

def init(node: Node):
    """
    Initialize new components needed for head_behavior:
    blackboard, dsd, subscribers
    """
    # This is a general purpose initialization function provided by moved
    # It is used to correctly initialize roscpp which is used in the collision checker module
    blackboard = HeadBlackboard(node)

    callback_group = ReentrantCallbackGroup()

    node.create_subscription(
        HeadModeMsg,
        'head_mode',
        blackboard.head_capsule.head_mode_callback,
        1,
        callback_group=callback_group)
    node.create_subscription(
        PoseWithCovarianceStamped,
        "ball_position_relative_filtered",
        blackboard.world_model.ball_filtered_callback,
        1,
        callback_group=callback_group)
    node.create_subscription(
        JointState,
        "joint_states",
        blackboard.head_capsule.joint_state_callback,
        1,
        callback_group=callback_group)
    blackboard.head_capsule.position_publisher = node.create_publisher(
        JointCommand,
        "head_motor_goals",
        10)
    blackboard.head_capsule.visual_compass_record_trigger = node.create_publisher(
        Header, blackboard.config['visual_compass_trigger_topic'], 5)

    dirname = get_package_share_directory("bitbots_head_behavior")

    dsd = DSD(blackboard, 'debug/dsd/head_behavior', node)
    dsd.register_actions(os.path.join(dirname, 'actions'))
    dsd.register_decisions(os.path.join(dirname, 'decisions'))
    dsd.load_behavior(os.path.join(dirname, 'head_behavior.dsd'))

    node.get_logger().debug("Head Behavior completely loaded")
    return dsd


def main(args=None):
    rclpy.init(args=None)
    node = Node("head_node", automatically_declare_parameters_from_overrides=True)
    dsd = init(node)
    node.create_timer(1 / 60.0, dsd.update)
    multi_executor = MultiThreadedExecutor(num_threads=10)
    multi_executor.add_node(node)


    try:
        multi_executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
