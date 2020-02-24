import rospy
from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class GoalScored(AbstractDecisionElement):
    def perform(self, reevaluate=False):
        if self.blackboard.gamestate.goal_update:
            self.blackboard.gamestate.goal_update = False
            return 'YES'
        else:
            return 'NO'

    def get_reevaluate(self):
        # Do not reevaluate, should only be reevaluated when cheering is done
        if self.blackboard.animation.is_animation_busy():
            return False
        else:
            return True
