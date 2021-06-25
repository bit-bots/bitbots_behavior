import numpy as np

from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class ReachedRolePosition(AbstractDecisionElement):
    def __init__(self, blackboard, dsd, parameters=None):
        super(ReachedRolePosition, self).__init__(blackboard, dsd, parameters)

        self.threshold = 0.3
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

        robot_position = self.blackboard.world_model.get_current_position()
        position = np.array([robot_position[0], robot_position[1]])
        if np.linalg.norm(goal - position) < self.threshold:
            return 'YES'
        return 'NO'

    def get_reevaluate(self):
        return True
