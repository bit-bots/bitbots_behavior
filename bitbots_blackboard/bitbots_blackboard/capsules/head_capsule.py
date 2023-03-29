import math
import numpy as np
from rclpy.publisher import Publisher
from rclpy.duration import Duration
from rclpy.node import Node
import tf2_ros as tf2

from humanoid_league_msgs.msg import HeadMode
from bitbots_moveit_bindings import check_collision
from bitbots_msgs.msg import JointCommand
from sensor_msgs.msg import JointState


class HeadCapsule:

    def __init__(self, blackboard):
        self.blackboard = blackboard

        # possible variables
        self.head_mode = None

        # preparing message for more performance
        self.pos_msg = JointCommand()
        self.pos_msg.joint_names = ["HeadPan", "HeadTilt"]
        self.pos_msg.positions = [0.0, 0.0]
        self.pos_msg.velocities = [0.0, 0.0]
        self.pos_msg.accelerations = [17.0, 17.0]
        self.pos_msg.max_currents = [-1.0, -1.0]

        self.position_publisher = None  # type: Publisher
        self.visual_compass_record_trigger = None  # type: Publisher

        self.tf_buffer = tf2.Buffer(Duration(seconds=5))
        # tf_listener is necessary, even though unused!
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self.blackboard.node)

        self.current_joint_state = JointState()
        self.current_joint_state.name = ["HeadPan", "HeadTilt"]
        self.current_joint_state.position = [0.0, 0.0]
        self.current_joint_state.velocity = [0.0, 0.0]
        self.current_joint_state.effort = [0.0, 0.0]

    def head_mode_callback(self, msg: HeadMode):
        """
        ROS Subscriber callback for /head_mode message.
        Saves the messages head mode on the blackboard
        """
        self.head_mode = msg.head_mode

    def joint_state_callback(self, msg):
        self.current_joint_state = msg


    #################
    # Head position #
    #################

    def _calculate_lower_speed(self, delta_fast_joint, delta_my_joint, speed):
        """
        Calculates the speed for the axis with the non maximum velocity.
        :param delta_fast_joint: The radians delta of the faster joint.
        :param delta_my_joint: The radians delta of the joint that should be slowed down.
        :param speed: The speed of the faster joint.
        :return: Speed of this slower joint.
        """
        estimated_time = delta_fast_joint / speed
        # Prevent zero division if goal is reached
        if estimated_time != 0:
            return delta_my_joint / estimated_time
        else:
            return 0

    def send_motor_goals(self,
                         pan_position,
                         tilt_position,
                         pan_speed=1.5,
                         tilt_speed=1.5,
                         current_pan_position=None,
                         current_tilt_position=None,
                         clip=True,
                         resolve_collision=False):
        """
        :param pan_position: pan in radians
        :param tilt_position: tilt in radians
        :param pan_speed:
        :param tilt_speed:
        :param clip: clip the motor values at the maximum value. This should almost always be true.
        :param current_pan_position: Current pan joint state for better interpolation (only active if both joints are set).
        :param current_tilt_position: Current tilt joint state for better interpolation (only active if both joints are set).
        :return: False if the target position collides, True otherwise
        """
        self.blackboard.node.get_logger().debug("target pan/tilt: {}/{}".format(pan_position, tilt_position))

        if clip:
            pan_position, tilt_position = self.pre_clip(pan_position, tilt_position)

        # Check if we should use the better interpolation
        if current_pan_position and current_tilt_position:
            if resolve_collision:
                success = self.avoid_collision_on_path(pan_position, tilt_position, current_pan_position,
                                                       current_tilt_position, pan_speed, tilt_speed)
                if not success:
                    self.blackboard.node.get_logger().error("Unable to resolve head collision")
                return success
            else:
                self.move_head_to_position_with_speed_adjustment(pan_position, tilt_position, current_pan_position,
                                                                 current_tilt_position, pan_speed, tilt_speed)
                return True
        else:  # Passes the stuff through
            self.pos_msg.positions = float(pan_position), float(tilt_position)
            self.pos_msg.velocities = float(pan_speed), float(tilt_speed)
            self.pos_msg.header.stamp = self.blackboard.node.get_clock().now().to_msg()
            self.position_publisher.publish(self.pos_msg)
            return True

    def avoid_collision_on_path(self,
                                goal_pan,
                                goal_tilt,
                                current_pan,
                                current_tilt,
                                pan_speed,
                                tilt_speed,
                                max_depth=4,
                                depth=0):
        # Backup behavior if max recursion depth is reached
        if depth > max_depth:
            self.move_head_to_position_with_speed_adjustment(0, 0, current_pan, current_tilt, pan_speed, tilt_speed)
            return False

        # Calculate distance in the joint space
        distance = math.hypot(goal_pan - current_pan, goal_tilt - current_tilt)

        # Calculate step size
        step_count = int(distance / math.radians(3))

        # Calculate path
        pan_steps = np.linspace(current_pan, goal_pan, step_count)
        tilt_steps = np.linspace(current_tilt, goal_tilt, step_count)
        path = zip(pan_steps, tilt_steps)

        # Checks if we have collisions on our path
        if any(map(self.check_head_collision, path)) or self.check_head_collision((goal_pan, goal_tilt)):
            # Check if the problem is solved if we move our head up at the goal position
            return self.avoid_collision_on_path(goal_pan, goal_tilt + math.radians(10), current_pan, current_tilt,
                                                pan_speed, tilt_speed, max_depth, depth + 1)
        else:
            # Every thing is fine, we can send our motor goals
            self.move_head_to_position_with_speed_adjustment(goal_pan, goal_tilt, current_pan, current_tilt, pan_speed,
                                                             tilt_speed)
            return True

    def check_head_collision(self, head_joints):
        joint_state = JointState()
        joint_state.name = ["HeadPan", "HeadTilt"]
        joint_state.position = head_joints
        return check_collision(joint_state)

    def move_head_to_position_with_speed_adjustment(self, goal_pan, goal_tilt, current_pan, current_tilt, pan_speed,
                                                    tilt_speed):
        # Calculate the deltas
        delta_pan = abs(current_pan - goal_pan)
        delta_tilt = abs(current_tilt - goal_tilt)
        # Check which speed should be lowered to achieve better interpolation
        if delta_pan > delta_tilt:
            tilt_speed = self._calculate_lower_speed(delta_pan, delta_tilt, pan_speed)
        else:
            pan_speed = self._calculate_lower_speed(delta_tilt, delta_pan, tilt_speed)
        # Send new joint values
        self.pos_msg.positions = float(goal_pan), float(goal_tilt)
        self.pos_msg.velocities = float(pan_speed), float(tilt_speed)
        self.pos_msg.header.stamp = self.blackboard.node.get_clock().now().to_msg()
        self.position_publisher.publish(self.pos_msg)

    def pre_clip(self, pan, tilt):
        """
        Return clipped motor goals for each axis

        :param pan: The goal pan position
        :param tilt: The goal tilt position
        :return (new_pan, new_tilt): Clipped motor goals
        """
        max_pan = self.blackboard.config['max_pan']
        max_tilt = self.blackboard.config['max_tilt']
        new_pan = min(max(pan, min(max_pan)), max(max_pan))
        new_tilt = min(max(tilt, min(max_tilt)), max(max_tilt))
        return new_pan, new_tilt

    ##################
    # Head positions #
    ##################

    def get_head_position(self):
        head_pan = self.current_joint_state.position[self.current_joint_state.name.index("HeadPan")]
        head_tilt = self.current_joint_state.position[self.current_joint_state.name.index("HeadTilt")]
        return head_pan, head_tilt

    #####################
    # Pattern generator #
    #####################

    def _lineAngle(self, line, line_count, min_angle, max_angle):
        """
        Converts a scanline number to an tilt angle
        """
        delta = abs(min_angle - max_angle)
        steps = delta / (line_count - 1)
        value = steps * line + min_angle
        return value

    def _calculateHorizontalAngle(self, is_right, angle_right, angle_left):
        """
        The right/left position to an pan angle
        """
        if is_right:
            return angle_right
        else:
            return angle_left

    def _interpolatedSteps(self, steps, tilt, min_pan, max_pan):
        """
        Splits a scanline in a number of dedicated steps
        """
        if steps == 0:
            return []
        steps += 1
        delta = abs(min_pan - max_pan)
        step_size = delta / float(steps)
        output_points = list()
        for i in range(1, steps):
            value = int(i * step_size + min_pan)
            point = (value, tilt)
            output_points.append(point)
        return output_points

    def generate_pattern(self,
                         lineCount,
                         maxHorizontalAngleLeft,
                         maxHorizontalAngleRight,
                         maxVerticalAngleUp,
                         maxVerticalAngleDown,
                         reduce_last_scanline=1,
                         interpolation_steps=0):
        """
        :param lineCount: Number of scanlines
        :param maxHorizontalAngleLeft: maximum look left angle
        :param maxHorizontalAngleRight: maximum look right angle
        :param maxVerticalAngleUp: maximum upwards angle
        :param maxVerticalAngleDown: maximum downwards angle
        :param interpolation_steps: number of interpolation steps for each line
        :return: List of angles (Pan, Tilt)
        """
        keyframes = []
        # Init first state
        downDirection = False
        rightSide = False
        rightDirection = True
        line = lineCount - 1
        # Calculate number of keyframes
        iterations = max((2 * lineCount - 2) * 2, 2)

        for i in range(iterations):
            # Create keyframe
            currentPoint = (self._calculateHorizontalAngle(rightSide, maxHorizontalAngleRight, maxHorizontalAngleLeft),
                            self._lineAngle(line, lineCount, maxVerticalAngleDown, maxVerticalAngleUp))
            # Add keyframe
            keyframes.append(currentPoint)

            # Interpolate to next keyframe if we are moving horizontally
            if rightSide != rightDirection:
                interpolatedKeyframes = self._interpolatedSteps(interpolation_steps, currentPoint[1],
                                                                maxHorizontalAngleRight, maxHorizontalAngleLeft)
                if rightDirection:
                    interpolatedKeyframes.reverse()
                keyframes.extend(interpolatedKeyframes)

            # Next state
            # Switch side
            if rightSide != rightDirection:
                rightSide = rightDirection
            # Or go up/down
            elif rightSide == rightDirection:
                rightDirection = not rightDirection
                if line in [0, lineCount - 1]:
                    downDirection = not downDirection
                if downDirection:
                    line -= 1
                else:
                    line += 1

        # Reduce the with of the last scanline if wanted.
        for index, keyframe in enumerate(keyframes):
            if keyframe[1] == maxVerticalAngleDown:
                keyframes[index] = (keyframe[0] * reduce_last_scanline, maxVerticalAngleDown)

        return keyframes
