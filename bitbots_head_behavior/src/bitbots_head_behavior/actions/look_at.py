import math

import rospy
import tf2_ros as tf2
from bio_ik_msgs.msg import IKRequest, LookAtGoal
from geometry_msgs.msg import PointStamped, Point

from dynamic_stack_decider.abstract_action_element import AbstractActionElement


class AbstractLookAt(AbstractActionElement):

    def __init__(self, blackboard, dsd, parameters=None):
        super(AbstractLookAt, self).__init__(blackboard, dsd, parameters)

        self.head_tf_frame = self.blackboard.config['head_transform_frame']
        self.tf_buffer = tf2.Buffer(rospy.Duration(5))
        # tf_listener is necessary, even though unused!
        self.tf_listener = tf2.TransformListener(self.tf_buffer)
        self.bio_ik_request = IKRequest()

        # Service proxy for LookAt
        self.request = IKRequest()
        self.request.group_name = "Head"
        self.request.timeout.secs = 1
        self.request.attempts = 1
        self.request.approximate = True
        self.request.look_at_goals.append(LookAtGoal())
        self.request.look_at_goals[0].link_name = "head"
        self.request.look_at_goals[0].weight = 1
        self.request.look_at_goals[0].axis.x = 1

    def get_motor_goals_from_point(self, point):
        """Call the look at service to calculate head motor goals"""

        target = Point(point.x, point.y, point.z)
        self.request.look_at_goals[0].target = target
        response = self.blackboard.bio_ik(self.request).ik_response
        states = response.solution.joint_state
        return states.position[states.name.index('HeadPan')], states.position[states.name.index('HeadTilt')]

    def _look_at(self, point):
        """
        Look at a point which is relative to the robot.

        The points header.frame_id determines the transforms reference frame of this point

        :type point: PointStamped
        """
        # transform the points reference frame to be the head
        try:
            point = self.tf_buffer.transform(point, self.head_tf_frame, timeout=rospy.Duration(0.9))
        except tf2.LookupException as e:
            rospy.logerr('Could not find transform {}. Either it does not exist or '
                         'transform is not yet online.\n{}'.format(self.head_tf_frame, e))
            return
        except tf2.ConnectivityException as e:
            rospy.logerr('No connection to transform\n{}'.format(e))
            return
        except tf2.ExtrapolationException as e:
            rospy.logerr('No transform yet\n{}'.format(e))
            return

        head_pan, head_tilt = self.get_motor_goals_from_point(point.point)
        head_pan, head_tilt = self._fix_pan_tilt_values(head_pan, head_tilt)
        self.blackboard.head_capsule.send_motor_goals(head_pan, head_tilt)

    def _fix_pan_tilt_values(self, pan, tilt):
        """
        Unfortunately, the bio_ik_service makes the robot look at a wrong point. As research via the
        test_look_at.py script suggests, this deviation is approximately a factor of 1.4 for both pan and
        tilt values. This is not good, but I do not know the reason. Therefore, this has to be kept to look
        at the correct position.
        """
        return pan / 1.4, tilt / 1.4


class LookDirection(AbstractLookAt):
    class Directions:
        """All possible directions"""
        FORWARD = 'FORWARD'
        DOWN = 'DOWN'
        UP = 'UP'

    def __init__(self, blackboard, dsd, parameters=None):
        """
        :param parameters['direction']: One of the possible directions
        """
        AbstractLookAt.__init__(self, blackboard, dsd, parameters)

        # Assert that a valid direction was supplied
        assert parameters is not None, 'No direction specified in parameters (key="direction")'
        assert 'direction' in parameters, 'No direction specified in parameters (key="direction")'
        assert parameters['direction'] in dir(self.Directions), 'Direction {} not found'.format(parameters["direction"])

        # Save supplied direction
        self.direction = getattr(self.Directions, parameters['direction'])
        self.position_down = self.blackboard.config['look_down_position']
        self.position_up = self.blackboard.config['look_up_position']
        self.position_forward = self.blackboard.config['look_forward_position']

    def perform(self, reevaluate=False):
        """
        Look at the direction direction that was supplied to this element

        :param reevaluate: No effect here
        """
        self.publish_debug_data('direction', self.direction)

        if self.direction == self.Directions.DOWN:
            head_pan, head_tilt = self.position_down
        elif self.direction == self.Directions.UP:
            head_pan, head_tilt = self.position_up
        else:
            head_pan, head_tilt = self.position_forward

        head_pan = head_pan / 180.0 * math.pi
        head_tilt = head_tilt / 180.0 * math.pi

        self.blackboard.head_capsule.send_motor_goals(head_pan, head_tilt)