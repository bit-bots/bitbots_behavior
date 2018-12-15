from geometry_msgs.msg import PointStamped

from bitbots_head_behavior.actions.look_at import AbstractLookAt


class TrackBall(AbstractLookAt):
    """
    This action follows the seen ball so that it the camera always points towards it.
    We try to do this so that the ball doesnt get lost as easily
    """

    def __init__(self, dsd, blackboard, parameters=None):
        super(TrackBall, self).__init__(dsd, blackboard, parameters)

    def perform(self, reevaluate=False):
        """
        Call look_at to look at the point which our world-model determines to be the ball

        :param reevaluate: No effect here
        """

        # Construct target point from target direction
        point = PointStamped()
        point.header.frame_id = 'base_footprint'
        u, v = self.blackboard.world_model.get_ball_position_uv()
        point.point.x = u
        point.point.y = v
        point.point.z = 0

        # Call internal look-at to turn head to this point
        self._look_at(point)
