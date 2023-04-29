from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.time import Time

from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class BallSeen(AbstractDecisionElement):
    """
    Decides whether the ball is currently being seen.
    To be precise the decision checks if the ball is currently getting located by our world-model.
    """

    def __init__(self, blackboard, dsd, parameters=None):
        super(BallSeen, self).__init__(blackboard, dsd, parameters)
        self.ball_lost_time = Duration(seconds=self.blackboard.config['ball_lost_time'])

    def perform(self, reevaluate=False):
        """
        Check if the time we last saw the ball is now.
        "Now"-Tolerance can be configured through ball_lost_time.

        :param reevaluate: Has no effect
        """

        if self.blackboard.world_model.has_ball_been_seen_by_myself():
            return 'YES'
        return 'NO'

    def get_reevaluate(self):
        """
        True

        because we need to act immediately if the ball_seen state changes
        """
        return True
