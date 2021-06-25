import numpy as np

from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class ReachedRolePosition(AbstractDecisionElement):
    def __init__(self, blackboard, dsd, parameters=None):
        super(ReachedRolePosition, self).__init__(blackboard, dsd, parameters)

        self.threshould = parameters['thres']
        self.role_positions = self.blackboard.config['role_positions']

    def perform(self, reevaluate=False):
        """
        Determines whether we are near the role position
        :param reevaluate:
        :return:
        """
        if self.blackboard.blackboard.duty == 'goalie':
            goal = np.array(self.role_positions[self.blackboard.blackboard.duty])
        elif self.blackboard.blackboard.duty == 'defense':
            goal = np.array(self.role_positions[self.blackboard.blackboard.duty][self.role_positions['pos_number']])
        elif self.blackboard.blackboard.duty == 'offense':
            kickoff_type = 'active' if self.blackboard.gamestate.has_kickoff() else 'passive'
            goal = np.array(
                self.role_positions[self.blackboard.blackboard.duty][kickoff_type][self.role_positions['pos_number']])

        if goal is None or self.blackboard.pathfinding.current_pose is None:
            return "NO"

        position = np.array([self.blackboard.pathfinding.current_pose.pose.position.x,
                             self.blackboard.pathfinding.current_pose.pose.position.y])
        if np.linalg.norm(goal - position) < self.threshould:
            return 'YES'
        return 'NO'

    def get_reevaluate(self):
        return True
