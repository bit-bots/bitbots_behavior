import rospy

from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class InGoalSearchTime(AbstractDecisionElement):
    def __init__(self, blackboard, dsd, parameters=None):
        super(InGoalSearchTime, self).__init__(blackboard, dsd, parameters)
        self.outrun_goal_search_time = self.blackboard.config['goal_search_duration']
        self.start_searching_time = rospy.get_time()

    def perform(self, reevaluate=False):
        if rospy.get_time() - self.start_searching_time < self.outrun_goal_search_time:
            return 'YES'
        print("not in goal search time")
        return 'NO'

    def get_reevaluate(self):
        return True
