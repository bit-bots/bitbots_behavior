import math
import numpy as np
import rospy

from dynamic_stack_decider.abstract_action_element import AbstractActionElement


class ActiveVision(AbstractActionElement):
    """
    Loads and runs the active vision policy
    """

    def __init__(self, dsd, blackboard, parameters=None):
        super(ActiveVision, self).__init__(dsd, blackboard, parameters)
        # Load model


    def perform(self, reevaluate=False):
        """
        Estimates the next viewpoint based on the loaded mlp policy

        :param reevaluate: No effect here
        """

        # Get last ball position
        point = self.blackboard.world_model.get_ball_stamped_relative()
        ball_x, ball_y = point.point.x, point.point.y
        ball_conf = max(0, 1 - 0.1 * self.blackboard.world_model.ball_seen_time.to_sec())
        # Normalize
        ball_x = ball_x / self.blackboard.world_model.field_length + 0.5
        ball_y = ball_y / self.blackboard.world_model.field_width + 0.5

        # Get robot pose
        robot_x, robot_y, robot_theta = self.blackboard.world_model.get_current_position()
        # Normalize
        robot_x = robot_x / self.blackboard.world_model.field_length + 0.5
        robot_y = robot_y / self.blackboard.world_model.field_width + 0.5
        robot_theta_sin = (math.sin(robot_theta) + 1)/2
        robot_theta_cos = (math.cos(robot_theta) + 1)/2

        # Get neck joint positions
        current_head_pan, current_head_tilt = self.blackboard.head_capsule.get_head_position()
        # Normalize
        pan_limits = [math.radians(-90),math.radians(90)]
        current_head_pan = (current_head_pan - min(pan_limits)) / (max(pan_limits) - min(pan_limits))
        tilt_limits = [0,math.radians(60)]
        current_head_tilt =  (current_head_tilt - min(tilt_limits)) / (max(tilt_limits) - min(tilt_limits))

        # Phase
        phase = math.sin(rospy.Time.now().to_sec()/10)

        print(
            robot_x,
            robot_y,
            robot_theta_sin,
            robot_theta_cos,
            current_head_pan,
            current_head_tilt,
            phase,
            ball_x,
            ball_y,
            ball_conf)
