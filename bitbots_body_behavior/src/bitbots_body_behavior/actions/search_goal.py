from dynamic_stack_decider.abstract_action_element import AbstractActionElement
from humanoid_league_msgs.msg import HeadMode


class SearchGoal(AbstractActionElement):
    def __init__(self, blackboard, dsd, parameters=None):
        super(SearchGoal, self).__init__(blackboard, dsd, parameters)

    def perform(self, reevaluate=False):
        # TODO search goal in head behaviour
