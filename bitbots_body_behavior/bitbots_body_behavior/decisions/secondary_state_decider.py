from bitbots_blackboard.blackboard import BodyBlackboard

from dynamic_stack_decider.abstract_decision_element import \
    AbstractDecisionElement
from bitbots_msgs.msg import GameState


class SecondaryStateDecider(AbstractDecisionElement):
    """
    Decides in which secondary state the game is currently in. The mode of the secondary state is handled in the
    game controller receiver, so the behavior does ont need to deal with this.
    """
    blackboard: BodyBlackboard
    def __init__(self, blackboard, dsd, parameters=None):
        super(SecondaryStateDecider, self).__init__(blackboard, dsd, parameters)
        self.secondary_game_states = {
            0: 'NORMAL',
            1: 'PENALTYSHOOT',  # should not happen during halftime or extra time
            2: 'OVERTIME',
            3: 'TIMEOUT',
            4: 'DIRECT_FREEKICK',
            5: 'INDIRECT_FREEKICK',
            6: 'PENALTYKICK',
            7: 'CORNER_KICK',
            8: 'GOAL_KICK',
            9: 'THROW_IN',
        }

    def perform(self, reevaluate=False):
        state_number = self.blackboard.gamestate.get_secondary_state()
        # todo this is a temporary hack to make GUI work
        if state_number == GameState.STATE_NORMAL:
            return "NORMAL"
        elif state_number == GameState.STATE_PENALTYSHOOT:
            return "PENALTYSHOOT"
        elif state_number == GameState.STATE_OVERTIME:
            return "OVERTIME"
        elif state_number == GameState.STATE_TIMEOUT:
            return "TIMEOUT"
        elif state_number == GameState.STATE_DIRECT_FREEKICK:
            return "DIRECT_FREEKICK"
        elif state_number == GameState.STATE_INDIRECT_FREEKICK:
            return "INDIRECT_FREEKICK"
        elif state_number == GameState.STATE_PENALTYKICK:
            return "PENALTYKICK"
        elif state_number == GameState.STATE_CORNER_KICK:
            return "CORNER_KICK"
        elif state_number == GameState.STATE_GOAL_KICK:
            return "GOAL_KICK"
        elif state_number == GameState.STATE_THROW_IN:
            return "THROW_IN"

    def get_reevaluate(self):
        """
        Secondary game state can change during the game
        """
        return True


class SecondaryStateTeamDecider(AbstractDecisionElement):
    """
    Decides if our team or the other team is allowed to execute the secondary state.
    """
    blackboard: BodyBlackboard
    def __init__(self, blackboard, dsd, parameters=None):
        super(SecondaryStateTeamDecider, self).__init__(blackboard, dsd)
        self.team_id = self.blackboard.gamestate.get_team_id()

    def perform(self, reevaluate=False):
        state_number = self.blackboard.gamestate.get_secondary_state()
        # we have to handle penalty shoot differently because the message is strange
        if state_number == GameState.STATE_PENALTYSHOOT:
            if self.blackboard.gamestate.has_kickoff():
                return 'OUR'
            return 'OTHER'
        else:
            if self.blackboard.gamestate.get_secondary_team() == self.team_id:
                return 'OUR'
            return 'OTHER'

    def get_reevaluate(self):
        """
        Secondary state Team can change during the game
        """
        return True


class SecondaryStateModeDecider(AbstractDecisionElement):
    """
    Decides which mode in the secondary state we are.
    """
    blackboard: BodyBlackboard
    def __init__(self, blackboard, dsd, parameters=None):
        super(SecondaryStateModeDecider, self).__init__(blackboard, dsd)
        self.modes = {
            0: 'PREPARATION',
            1: 'PLACING',  # should not happen during halftime or extra time
            2: 'END',
        }

    def perform(self, reevaluate=False):
        state_mode = self.blackboard.gamestate.get_secondary_state_mode()
        return self.modes[state_mode]

    def get_reevaluate(self):
        return True
