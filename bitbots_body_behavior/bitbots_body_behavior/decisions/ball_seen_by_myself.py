from bitbots_blackboard.blackboard import BodyBlackboard
from rclpy.duration import Duration

from dynamic_stack_decider.abstract_decision_element import \
    AbstractDecisionElement


class BallSeenByMyself(AbstractDecisionElement):
    blackboard: BodyBlackboard
    def __init__(self, blackboard, dsd, parameters=None):
        super(BallSeenByMyself, self).__init__(blackboard, dsd, parameters)
        self.ball_lost_time = Duration(seconds=self.blackboard.config['ball_lost_time'])

    def perform(self, reevaluate=False):
        """Determines whether the ball was seen recently by myself (as defined in config)"""
        self.publish_debug_data("Time since ball last seen by myself",
                                self.blackboard.node.get_clock().now() - self.blackboard.world_model.get_time_ball_last_seen_by_myself())
        if self.blackboard.world_model.has_ball_been_seen_by_myself():
            return 'YES'
        return 'NO'

    def get_reevaluate(self):
        return True
