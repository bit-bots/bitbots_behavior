import math
from typing import Dict, List, Optional, Tuple

from bitbots_utils.utils import get_parameters_from_other_node
from geometry_msgs.msg import PointStamped, Pose
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.time import Time as RclpyTime
from builtin_interfaces.msg import Time as MsgTime
from std_msgs.msg import Float32, Header

from humanoid_league_msgs.msg import Strategy, TeamData


class TeamDataCapsule:

    def __init__(self, node: Node):
        self.node = node

        # Retrieve game settings from parameter blackboard
        params = get_parameters_from_other_node(self.node, 'parameter_blackboard', ['bot_id', 'role'])
        self.bot_id = params['bot_id']
        role_name = params['role']

        self.strategy_sender: Optional[Publisher] = None
        self.time_to_ball_publisher: Optional[Publisher] = None
        # indexed with one to match robot ids
        self.team_data : Dict[int, TeamData] = {}
        for i in range(1, 7):
            self.team_data[i] = TeamData()
        self.roles = {
            'striker': Strategy.ROLE_STRIKER,
            'offense': Strategy.ROLE_STRIKER,
            'supporter': Strategy.ROLE_SUPPORTER,
            'defender': Strategy.ROLE_DEFENDER,
            'defense': Strategy.ROLE_DEFENDER,
            'other': Strategy.ROLE_OTHER,
            'goalie': Strategy.ROLE_GOALIE,
            'idle': Strategy.ROLE_IDLING
        }
        self.own_time_to_ball = 9999.0
        self.strategy = Strategy()
        self.strategy.role = self.roles[role_name]
        self.strategy_update: float = 0.0
        self.action_update: float = 0.0
        self.role_update: float = 0.0
        self.map_frame: str = self._blackboard.node.get_parameter('map_frame').value
        self.teammate_ball = PointStamped()
        self.teammate_ball_default_header = Header(frame_id=self.map_frame, stamp=RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME).to_msg())
        self.teammate_ball.header = self.teammate_ball_default_header
        self.data_timeout: float = self.node.get_parameter("team_data_timeout").value
        self.ball_max_covariance: float  = self.node.get_parameter("ball_max_covariance").value
        self.ball_lost_time: float = Duration(seconds=self.node.get_parameter('body.ball_lost_time').value)
        self.localization_precision_threshold_x_sdev: float = self.node.get_parameter(
            'body.localization_precision_threshold.x_sdev').value
        self.localization_precision_threshold_y_sdev: float = self.node.get_parameter(
            'body.localization_precision_threshold.y_sdev').value
        self.localization_precision_threshold_theta_sdev: float = self.node.get_parameter(
            'body.localization_precision_threshold.theta_sdev').value

    def is_valid(self, data: TeamData) -> bool:
        return self.node.get_clock().now() - RclpyTime.from_msg(data.header.stamp) < Duration(seconds=self.data_timeout) \
               and data.state != TeamData.STATE_PENALIZED

    def get_goalie_ball_position(self) -> Optional[Tuple[float, float]]:
        """Return the ball relative to the goalie

        :return a tuple with the relative ball and the last update time
        """
        data: TeamData
        for data in self.team_data.values():
            role = data.strategy.role
            if role == Strategy.ROLE_GOALIE and self.is_valid(data):
                return data.ball_relative.pose.position.x, data.ball_relative.pose.position.y
        return None

    def get_goalie_ball_distance(self) -> Optional[float]:
        """Return the distance between the goalie and the ball

        :return a tuple with the ball-goalie-distance and the last update time
        """
        goalie_ball_position = self.get_goalie_ball_position()
        if goalie_ball_position is not None:
            return math.hypot(goalie_ball_position[0],goalie_ball_position[1])
        else:
            return None

    def is_goalie_handling_ball(self):
        """ Returns true if the goalie is going to the ball."""
        data: TeamData
        for data in self.team_data.values():
            if self.is_valid(data) \
                    and data.strategy.role == Strategy.ROLE_GOALIE \
                    and data.strategy.action in [Strategy.ACTION_GOING_TO_BALL, Strategy.ACTION_KICKING]:
                return True
        return False

    def is_team_mate_kicking(self):
        """Returns true if one of the players in the own team is kicking."""
        data: TeamData
        for data in self.team_data.values():
            if self.is_valid(data) and data.strategy.action == Strategy.ACTION_KICKING:
                return True
        return False

    def team_rank_to_ball(self, own_ball_distance: float, count_goalies: bool = True, use_time_to_ball: bool = False):
        """
        Returns the rank of this robot compared to the team robots concerning ball distance.

        Ignores the goalies distance, as it should not leave the goal, even if it is closer than field players.
        For example, we do not want our goalie to perform a throw in against our empty goal.

        :return the rank from 1 (nearest) to the number of robots
        """
        distances = []
        data: TeamData
        for data in self.team_data.values():
            # data should not be outdated, from a robot in play, only goalie if desired,
            # x and y covariance values should be below threshold. orientation covariance of ball does not matter
            # covariance is a 6x6 matrix as array. 0 is x, 7 is y
            if self.is_valid(data) and (
                    data.strategy.role != Strategy.ROLE_GOALIE or count_goalies) \
                    and data.ball_absolute.covariance[0] < self.ball_max_covariance \
                    and data.ball_absolute.covariance[7] < self.ball_max_covariance:
                if use_time_to_ball:
                    distances.append(data.time_to_position_at_ball)
                else:
                    distances.append(self.get_robot_ball_euclidean_distance(data))
        for rank, distance in enumerate(sorted(distances)):
            if own_ball_distance < distance:
                return rank + 1
        return len(distances) + 1

    def get_robot_ball_euclidean_distance(self, robot_teamdata: TeamData):
        ball_rel_x = robot_teamdata.ball_absolute.pose.position.x - robot_teamdata.robot_position.pose.position.x
        ball_rel_y = robot_teamdata.ball_absolute.pose.position.y - robot_teamdata.robot_position.pose.position.y
        dist = math.hypot(ball_rel_x, ball_rel_y)
        return dist

    def set_role(self, role: int):
        """Set the role of this robot in the team

        :param role: Has to be a role from humanoid_league_msgs/Strategy
        """
        assert role in [
            Strategy.ROLE_STRIKER, Strategy.ROLE_SUPPORTER, Strategy.ROLE_DEFENDER, Strategy.ROLE_OTHER,
            Strategy.ROLE_GOALIE, Strategy.ROLE_IDLING
        ]
        self.strategy.role = role
        self.role_update = float(self.node.get_clock().now().seconds_nanoseconds()[0] +
                                 self.node.get_clock().now().seconds_nanoseconds()[1] / 1e9)

    def get_role(self) -> Tuple[int, float]:
        return self.strategy.role, self.role_update

    def set_action(self, action: int):
        """Set the action of this robot

        :param action: An action from humanoid_league_msgs/Strategy"""
        assert action in [
            Strategy.ACTION_UNDEFINED, Strategy.ACTION_POSITIONING, Strategy.ACTION_GOING_TO_BALL,
            Strategy.ACTION_TRYING_TO_SCORE, Strategy.ACTION_WAITING, Strategy.ACTION_SEARCHING,
            Strategy.ACTION_KICKING, Strategy.ACTION_LOCALIZING
        ]
        self.strategy.action = action
        self.action_update = float(self.node.get_clock().now().seconds_nanoseconds()[0] +
                                   self.node.get_clock().now().seconds_nanoseconds()[1] / 1e9)

    def get_action(self) -> Tuple[int, float]:
        return self.strategy.action, self.action_update

    def set_kickoff_strategy(self, strategy: int):
        assert strategy in [Strategy.SIDE_LEFT, Strategy.SIDE_MIDDLE, Strategy.SIDE_RIGHT]
        self.strategy.offensive_side = strategy
        self.strategy_update = float(self.node.get_clock().now().seconds_nanoseconds()[0] +
                                     self.node.get_clock().now().seconds_nanoseconds()[1] / 1e9)

    def get_kickoff_strategy(self) -> Tuple[int, float]:
        return self.strategy.offensive_side, self.strategy_update

    def get_active_teammate_poses(self, count_goalies: bool = False) -> List[Pose]:
        """ Returns the poses of all playing robots """
        poses = []
        data: TeamData
        for data in self.team_data.values():
            if self.is_valid(data) and (data.strategy.role != Strategy.ROLE_GOALIE or count_goalies):
                poses.append(data.robot_position.pose)
        return poses

    def set_own_time_to_ball(self, time_to_ball: float):
        self.own_time_to_ball = time_to_ball

    def get_own_time_to_ball(self) -> float:
        return self.own_time_to_ball

    def publish_time_to_ball(self):
        self.time_to_ball_publisher.publish(Float32(data=self.get_own_time_to_ball()))

    def team_data_callback(self, msg: TeamData):
        # Save team data
        self.team_data[msg.robot_id] = msg
        self.teammate_ball = self._get_best_teammate_ball()

    def publish_strategy(self):
        """Publish for team comm"""
        self.strategy_sender.publish(self.strategy)

    def get_teammate_ball_seen_time(self) -> RclpyTime:
        """Returns the time at which a teammate has seen the ball accurately enough"""
        teammate_ball = self.get_teammate_ball()
        if teammate_ball is not None:
            return RclpyTime.from_msg(teammate_ball.header.stamp)
        else:
            return RclpyTime(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME)

    def teammate_ball_is_valid(self):
        """Returns true if a teammate has seen the ball accurately enough"""
        return self.teammate_ball is not None

    def get_teammate_ball(self) -> Optional[PointStamped]:
        """Returns the ball from the closest teammate that has accurate enough localization and ball precision"""
        return self.teammate_ball

    def _get_best_teammate_ball(self) -> Optional[PointStamped]:
        """Returns the ball from the closest teammate that has accurate enough localization and ball precision"""

        def std_dev_from_covariance(covariance):
            x_sdev = covariance[0]  # position 0,0 in a 6x6-matrix
            y_sdev = covariance[7]  # position 1,1 in a 6x6-matrix
            theta_sdev = covariance[35]  # position 5,5 in a 6x6-matrix
            return x_sdev, y_sdev, theta_sdev

        best_robot_dist = 9999
        best_ball = None

        teamdata: TeamData
        for teamdata in self.team_data.values():
            if not self.is_valid(teamdata):
                continue
            ball = teamdata.ball_absolute
            ball_x_std_dev, ball_y_std_dev, _ = std_dev_from_covariance(ball.covariance)
            robot = teamdata.robot_position
            robot_x_std_dev, robot_y_std_dev, robot_theta_std_dev = std_dev_from_covariance(robot.covariance)
            stamp = teamdata.header.stamp
            if self.node.get_clock().now() - RclpyTime.from_msg(stamp) < self.ball_lost_time:
                if ball_x_std_dev < self.ball_max_covariance and ball_y_std_dev < self.ball_max_covariance:
                    if robot_x_std_dev < self.localization_precision_threshold_x_sdev and \
                            robot_y_std_dev < self.localization_precision_threshold_y_sdev and \
                            robot_theta_std_dev < self.localization_precision_threshold_theta_sdev:
                        robot_dist = self.get_robot_ball_euclidean_distance(teamdata)
                        if robot_dist < best_robot_dist:
                            best_ball = PointStamped()
                            best_ball.header = teamdata.header
                            best_ball.point.x = teamdata.ball_absolute.pose.position.x
                            best_ball.point.y = teamdata.ball_absolute.pose.position.y
                            best_robot_dist = robot_dist
        return best_ball

    def forget_teammate_ball(self):
        self.teammate_ball = PointStamped()
        self.teammate_ball.header = self.teammate_ball_default_header
