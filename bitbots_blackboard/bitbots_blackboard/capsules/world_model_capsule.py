import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from bitbots_blackboard.blackboard import BodyBlackboard

import numpy as np
import tf2_ros as tf2
from bitbots_utils.utils import (get_parameter_dict,
                                 get_parameters_from_other_node)
from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped, TwistStamped,
                               TwistWithCovarianceStamped)
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
from builtin_interfaces.msg import Time as MsgTime
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_geometry_msgs import PointStamped
from tf_transformations import euler_from_quaternion


class WorldModelCapsule:
    def __init__(self, blackboard: "BodyBlackboard"):
        self._blackboard = blackboard

        self.tf_buffer = tf2.Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self._blackboard.node)

        # Parameters
        parameters = get_parameters_from_other_node(
            self._blackboard.node,
            "/parameter_blackboard",
            ["field_length", "field_width", "goal_width"])
        self.field_length: float = parameters["field_length"]
        self.field_width: float = parameters["field_width"]
        self.goal_width: float = parameters["goal_width"]

        self.base_footprint_frame: str = self._blackboard.node.get_parameter('base_footprint_frame').value
        self.map_frame: str = self._blackboard.node.get_parameter('map_frame').value
        self.odom_frame: str = self._blackboard.node.get_parameter('odom_frame').value

        self.obstacle_costmap_smoothing_sigma: float = self._blackboard.node.get_parameter('body.obstacle_costmap_smoothing_sigma').value
        self.ball_twist_precision_threshold: Dict[str, float] = get_parameter_dict(self._blackboard.node, 'body.ball_twist_precision_threshold')
        self.localization_precision_threshold: Dict[str, float] = get_parameter_dict(self._blackboard.node,'body.localization_precision_threshold')
        self.body_config: Dict[str, float] = get_parameter_dict(self._blackboard.node, "body")
        self.ball_lost_time = Duration(seconds=self.body_config['ball_lost_time'])
        self.ball_twist_lost_time = Duration(seconds=self.body_config['ball_twist_lost_time'])
        self.map_margin: float = self.body_config['map_margin']
        self.obstacle_cost: float = self.body_config['obstacle_cost']

        # Placeholders
        self.pose = PoseWithCovarianceStamped()  # Own pose with covariance
        self._ball_seen: bool = False
        self._ball_seen_time: RclpyTime = RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME)
        self.ball_base_footprint = PointStamped()  # The ball in the base footprint frame
        self.ball_base_footprint_default_header = Header(frame_id=self.base_footprint_frame, stamp=RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME).to_msg())
        self.ball_base_footprint.header = self.ball_base_footprint_default_header
        self.ball_odom = PointStamped()  # The ball in the odom frame (when localization is not usable)
        self.ball_odom_default_header = Header(frame_id=self.odom_frame, stamp=RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME).to_msg())
        self.ball_odom.header = self.ball_odom_default_header
        self.ball_map = PointStamped()  # The ball in the map frame (when localization is usable)
        self.ball_map_default_header = Header(frame_id=self.map_frame, stamp=RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME).to_msg())
        self.ball_map.header = self.ball_map_default_header
        self.ball_twist_map: Optional[TwistStamped] = None
        self.ball_filtered: Optional[PoseWithCovarianceStamped] = None

        # Publisher for visualization in RViZ
        self.debug_publisher_ball = self._blackboard.node.create_publisher(PointStamped, 'debug/viz_ball', 1)
        self.debug_publisher_ball_twist = self._blackboard.node.create_publisher(TwistStamped, 'debug/ball_twist', 1)
        self.debug_publisher_used_ball = self._blackboard.node.create_publisher(PointStamped, 'debug/used_ball', 1)
        self.debug_publisher_which_ball = self._blackboard.node.create_publisher(Header, 'debug/which_ball_is_used', 1)

        self.reset_ball_filter_client = self._blackboard.node.create_client(Trigger, 'ball_filter_reset')


    ############
    ### Ball ###
    ############

    def get_time_ball_last_seen_by_myself(self) -> RclpyTime:
        """Returns the time at which I myself have seen the ball last recently"""
        return self._ball_seen_time

    def has_ball_been_seen_by_myself(self) -> bool:
        """Returns true if I myself have seen the ball recently (less than ball_lost_time ago)"""
        elapsed_time: Duration = self._blackboard.node.get_clock().now() - self.get_time_ball_last_seen_by_myself()
        return elapsed_time < self.ball_lost_time

    def get_time_ball_last_seen_by_myself_or_team(self) -> RclpyTime:
        """
        Returns the time at which the ball was last seen by either myself or a teammate (if available and localized).
        """
        if hasattr(self._blackboard, "team_data") and self.localization_precision_in_threshold():
            # Team data available and localization usable
            # Compare times of ball seen by myself and teammate, return the latest
            return max(self.get_time_ball_last_seen_by_myself(), self._blackboard.team_data.get_teammate_ball_seen_time())
        else:
            # Team data not available or localization not usable
            # Return time of ball seen by myself
            return self.get_time_ball_last_seen_by_myself()

    def has_ball_been_seen_by_myself_or_team(self) -> bool:
        """Returns true if I myself or a teammate have seen the ball recently (less than ball_lost_time ago)"""
        elapsed_time: Duration = self._blackboard.node.get_clock().now() - self.get_time_ball_last_seen_by_myself_or_team()
        return elapsed_time < self.ball_lost_time

    def get_best_ball(self) -> PointStamped:
        """
        Returns the best ball.
        """
        best_ball: PointStamped
        debug_frame_id: str

        if self.localization_precision_in_threshold():  # Localization usable
            if (
                self.has_ball_been_seen_by_myself() or  # I myself have seen the ball recently
                not hasattr(self._blackboard, "team_data")  # OR Team data not available
            ):
                best_ball = self.ball_map
                debug_frame_id = "own_ball_map"
            else:  # I myself have not seen the ball recently but team data is available
                teammate_ball = self._blackboard.team_data.get_teammate_ball()
                if (
                    self._blackboard.team_data.teammate_ball_is_valid() and  # Teammate has seen the ball recently and is accurate enough
                    self.tf_buffer.can_transform(  # AND the ball can be transformed
                        self.base_footprint_frame,
                        teammate_ball.header.frame_id,
                        teammate_ball.header.stamp,
                        timeout=Duration(seconds=0.2))
                ):
                    best_ball = teammate_ball
                    debug_frame_id = "teammate_ball"
                else:  # Teammate has not seen the ball recently or is not accurate enough or the ball cannot be transformed
                    self._blackboard.node.get_logger().warning(
                        "My ball is bad but the teammates ball is worse or cant be transformed")
                    # Fall back to my own ball
                    best_ball = self.ball_map
                    debug_frame_id = "own_ball_map"
        else:  # Localization not usable
            best_ball = self.ball_odom
            debug_frame_id = "own_ball_odom"

        # Publish debug information
        header = Header()
        header.stamp = best_ball.header.stamp
        header.frame_id = debug_frame_id
        self.debug_publisher_which_ball.publish(header)
        self.debug_publisher_used_ball.publish(best_ball)
        return best_ball

    def get_ball_position_xy(self) -> Tuple[float, float]:
        """Return the ball saved in the map or odom frame"""
        ball = self.get_best_ball()
        return ball.point.x, ball.point.y

    def get_ball_position_uv(self) -> Tuple[float, float]:
        ball = self.get_best_ball()
        try:
            ball_bfp = self.tf_buffer.transform(ball, self.base_footprint_frame, timeout=Duration(seconds=0.2)).point
        except (tf2.ExtrapolationException) as e:
            self._blackboard.node.get_logger().warn(e)
            self._blackboard.node.get_logger().error('Severe transformation problem concerning the ball!')
            return None
        return ball_bfp.x, ball_bfp.y

    def get_ball_distance(self) -> float:
        ball_pos = self.get_ball_position_uv()
        if ball_pos is None:
            return np.inf  # worst case (very far away)
        else:
            u, v = ball_pos
        return math.hypot(u, v)

    def get_ball_angle(self) -> float:
        ball_pos = self.get_ball_position_uv()
        if ball_pos is None:
            return -math.pi  # worst case (behind robot)
        else:
            u, v = ball_pos
        return math.atan2(v, u)

    def ball_filtered_callback(self, msg: PoseWithCovarianceStamped):
        self.ball_filtered = msg

        # When the precision is not sufficient, the ball ages.
        x_sdev = msg.pose.covariance[0]  # position 0,0 in a 6x6-matrix
        y_sdev = msg.pose.covariance[7]  # position 1,1 in a 6x6-matrix
        if x_sdev > self.body_config['ball_position_precision_threshold']['x_sdev'] or \
                y_sdev > self.body_config['ball_position_precision_threshold']['y_sdev']:
            self.forget_ball(own=True, team=False, reset_ball_filter=False)
            return

        ball_buffer = PointStamped(header=msg.header, point=msg.pose.pose.position)
        try:
            self.ball_base_footprint = self.tf_buffer.transform(ball_buffer, self.base_footprint_frame, timeout=Duration(seconds=1.0))
            self.ball_odom = self.tf_buffer.transform(ball_buffer, self.odom_frame, timeout=Duration(seconds=1.0))
            self.ball_map = self.tf_buffer.transform(ball_buffer, self.map_frame, timeout=Duration(seconds=1.0))

            # Set timestamps to zero to get the newest transform when this is transformed later
            # TODO: Why? This sounds fishy
            self.ball_odom.header.stamp: MsgTime = RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME).to_msg()
            self.ball_map.header.stamp: MsgTime = RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME).to_msg()

            self._ball_seen_time = RclpyTime.from_msg(msg.header.stamp)
            self._ball_seen = True
            self.debug_publisher_ball.publish(self.ball_base_footprint)

        except (tf2.ConnectivityException, tf2.LookupException, tf2.ExtrapolationException) as e:
            self._blackboard.node.get_logger().warn(str(e))

    def recent_ball_twist_available(self) ->  bool:
        if self.ball_twist_map is None:
            return False
        elapsed_time: Duration = self._blackboard.node.get_clock().now() - self.ball_twist_map.header.stamp
        return elapsed_time < self.ball_twist_lost_time

    def ball_twist_callback(self, msg: TwistWithCovarianceStamped):
        x_sdev = msg.twist.covariance[0]  # position 0,0 in a 6x6-matrix
        y_sdev = msg.twist.covariance[7]  # position 1,1 in a 6x6-matrix
        if x_sdev > self.ball_twist_precision_threshold['x_sdev'] or \
                y_sdev > self.ball_twist_precision_threshold['y_sdev']:
            return
        if msg.header.frame_id != self.map_frame:
            try:
                # point (0,0,0)
                point_a = PointStamped()
                point_a.header = msg.header
                # linear velocity vector
                point_b = PointStamped()
                point_b.header = msg.header
                point_b.point.x = msg.twist.twist.linear.x
                point_b.point.y = msg.twist.twist.linear.y
                point_b.point.z = msg.twist.twist.linear.z
                # transform start and endpoint of velocity vector
                point_a = self.tf_buffer.transform(point_a, self.map_frame, timeout=Duration(seconds=1.0))
                point_b = self.tf_buffer.transform(point_b, self.map_frame, timeout=Duration(seconds=1.0))
                # build new twist using transform vector
                self.ball_twist_map = TwistStamped(header=msg.header)
                self.ball_twist_map.header.frame_id = self.map_frame
                self.ball_twist_map.twist.linear.x = point_b.point.x - point_a.point.x
                self.ball_twist_map.twist.linear.y = point_b.point.y - point_a.point.y
                self.ball_twist_map.twist.linear.z = point_b.point.z - point_a.point.z
            except (tf2.ConnectivityException, tf2.LookupException, tf2.ExtrapolationException) as e:
                self._blackboard.node.get_logger().warn(str(e))
        else:
            self.ball_twist_map = TwistStamped(header=msg.header, twist=msg.twist.twist)
        if self.ball_twist_map is not None:
            self.debug_publisher_ball_twist.publish(self.ball_twist_map)

    def forget_ball(self, own: bool = True, team: bool = True, reset_ball_filter: bool = True) -> None:
        """
        Forget that we and the best teammate saw a ball, optionally reset the ball filter
        :param own: Forget the ball recognized by the own robot, defaults to True
        :type own: bool, optional
        :param team: Forget the ball received from the team, defaults to True
        :type team: bool, optional
        :param reset_ball_filter: Reset the ball filter, defaults to True
        :type reset_ball_filter: bool, optional
        """
        if own:  # Forget own ball
            self._ball_seen_time = RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME)
            self.ball_base_footprint = PointStamped()
            self.ball_base_footprint.header = self.ball_base_footprint_default_header

        if team:  # Forget team ball
            self._blackboard.team_data.forget_teammate_ball()

        if reset_ball_filter:  # Reset the ball filter
            result: Trigger.Response = self.reset_ball_filter_client.call(Trigger.Request())
            if result.success:
                self._blackboard.node.get_logger().info(f"Received message from ball filter: '{result.message}'")
            else:
                self._blackboard.node.get_logger().warn(f"Ball filter reset failed with: '{result.message}'")

    ########
    # Goal #
    ########

    def get_map_based_opp_goal_center_uv(self):
        x, y = self.get_map_based_opp_goal_center_xy()
        return self.get_uv_from_xy(x, y)

    def get_map_based_opp_goal_center_xy(self):
        return self.field_length / 2, 0.0

    def get_map_based_own_goal_center_uv(self):
        x, y = self.get_map_based_own_goal_center_xy()
        return self.get_uv_from_xy(x, y)

    def get_map_based_own_goal_center_xy(self):
        return -self.field_length / 2, 0.0

    def get_map_based_opp_goal_angle_from_ball(self):
        ball_x, ball_y = self.get_ball_position_xy()
        goal_x, goal_y = self.get_map_based_opp_goal_center_xy()
        return math.atan2(goal_y - ball_y, goal_x - ball_x)

    def get_map_based_opp_goal_distance(self):
        x, y = self.get_map_based_opp_goal_center_xy()
        return self.get_distance_to_xy(x, y)

    def get_map_based_opp_goal_angle(self):
        x, y = self.get_map_based_opp_goal_center_uv()
        return math.atan2(y, x)

    def get_map_based_opp_goal_left_post_uv(self):
        x, y = self.get_map_based_opp_goal_center_xy()
        return self.get_uv_from_xy(x, y - self.goal_width / 2)

    def get_map_based_opp_goal_right_post_uv(self):
        x, y = self.get_map_based_opp_goal_center_xy()
        return self.get_uv_from_xy(x, y + self.goal_width / 2)

    ########
    # Pose #
    ########

    def pose_callback(self, pose: PoseWithCovarianceStamped):
        self.pose = pose

    def get_current_position(self) -> Tuple[float, float, float]:
        """
        Returns the current position as determined by the localization
        :returns x,y,theta
        """
        transform = self.get_current_position_transform()
        if transform is None:
            return None
        orientation = transform.transform.rotation
        theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
        return transform.transform.translation.x, transform.transform.translation.y, theta

    def get_current_position_pose_stamped(self) -> PoseStamped:
        """
        Returns the current position as determined by the localization as a PoseStamped
        """
        transform = self.get_current_position_transform()
        if transform is None:
            return None
        ps = PoseStamped()
        ps.header = transform.header
        ps.pose.position.x = transform.transform.translation.x
        ps.pose.position.y = transform.transform.translation.y
        ps.pose.position.z = transform.transform.translation.z
        ps.pose.orientation = transform.transform.rotation
        return ps

    def get_current_position_transform(self) -> TransformStamped:
        """
        Returns the current position as determined by the localization as a TransformStamped
        """
        try:
            # get the most recent transform
            transform = self.tf_buffer.lookup_transform(self.map_frame, self.base_footprint_frame,
                                                        RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME))
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            self._blackboard.node.get_logger().warn(str(e))
            return None
        return transform

    def get_localization_precision(self) -> Tuple[float, float, float]:
        """
        Returns the current localization precision based on the covariance matrix.
        """
        x_sdev = self.pose.pose.covariance[0]  # position 0,0 in a 6x6-matrix
        y_sdev = self.pose.pose.covariance[7]  # position 1,1 in a 6x6-matrix
        theta_sdev = self.pose.pose.covariance[35]  # position 5,5 in a 6x6-matrix
        return (x_sdev, y_sdev, theta_sdev)

    def localization_precision_in_threshold(self) -> bool:
        """
        Returns whether the last localization precision values were in the threshold defined in the settings.
        """
        # Check whether we can transform into and from the map frame seconds.
        if not self.localization_pose_current():
            return False
        # get the standard deviation values of the covariance matrix
        precision = self.get_localization_precision()
        # return whether those values are in the threshold
        return precision[0] < self.localization_precision_threshold['x_sdev'] and \
               precision[1] < self.localization_precision_threshold['y_sdev'] and \
               precision[2] < self.localization_precision_threshold['theta_sdev']

    def localization_pose_current(self) -> bool:
        """
        Returns whether we can transform into and from the map frame.
        """
        # if we can do this, we should be able to transform the ball
        # (unless the localization dies in the next 0.2 seconds)
        try:
            t = self._blackboard.node.get_clock().now() - Duration(seconds=0.3)
        except TypeError as e:
            self._blackboard.node.get_logger().error(e)
            t = RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME)
        return self.tf_buffer.can_transform(self.base_footprint_frame, self.map_frame, t)

    ##########
    # Common #
    ##########

    def get_uv_from_xy(self, x, y) -> Tuple[float, float]:
        """ Returns the relativ positions of the robot to this absolute position"""
        current_position = self.get_current_position()
        x2 = x - current_position[0]
        y2 = y - current_position[1]
        theta = -1 * current_position[2]
        u = math.cos(theta) * x2 + math.sin(theta) * y2
        v = math.cos(theta) * y2 - math.sin(theta) * x2
        return u, v

    def get_xy_from_uv(self, u, v):
        """ Returns the absolute position from the given relative position to the robot"""
        pos_x, pos_y, theta = self.get_current_position()
        angle = math.atan2(v, u) + theta
        hypotenuse = math.hypot(u, v)
        return pos_x + math.sin(angle) * hypotenuse, pos_y + math.cos(angle) * hypotenuse

    def get_distance_to_xy(self, x, y):
        """ Returns distance from robot to given position """
        u, v = self.get_uv_from_xy(x, y)
        dist = math.hypot(u, v)
        return dist
