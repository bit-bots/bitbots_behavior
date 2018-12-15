import rospy
from humanoid_league_msgs.msg import HeadMode as HeadModeMsg
from bitbots_msgs.msg import JointCommand


class HeadCapsule:
    def __init__(self, blackboard):
        self.blackboard = blackboard

        # possible variables
        self.head_mode = None

        # preparing message for more performance
        self.pos_msg = JointCommand()
        self.pos_msg.joint_names = ["HeadPan", "HeadTilt"]
        self.pos_msg.positions = [0, 0]
        self.pos_msg.velocities = [0, 0]
        self.pos_msg.accelerations = [-1, -1]
        self.pos_msg.max_currents = [-1, -1]

        self.position_publisher = None  # type: rospy.Publisher

    def head_mode_callback(self, msg):
        """
        ROS Subscriber callback for /head_mode message.
        Saves the messages head mode on the blackboard
        """
        self.head_mode = msg.headMode

    #################
    # Head position #
    #################

    def send_motor_goals(self, pan_position, tilt_position, pan_speed=1.5, tilt_speed=1.5):
        """
        :param pan_position: pan in radians
        :param tilt_position: tilt in radians
        :param pan_speed:
        :param tilt_speed:
        :return:
        """
        # 3 is slower than maximum, maybe it is good
        print('{}/{}'.format(pan_position, tilt_position))
        pan_position = min(max(pan_position, -1), 1)
        tilt_position = min(max(tilt_position, -1.2), 0)
        self.pos_msg.positions = pan_position, tilt_position
        self.pos_msg.velocities = [pan_speed, tilt_speed]
        self.pos_msg.header.stamp = rospy.Time.now()
        self.position_publisher.publish(self.pos_msg)
