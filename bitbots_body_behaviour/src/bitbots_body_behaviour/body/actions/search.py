"""
Search
^^^^^^
"""
from bitbots_stackmachine.abstract_action_module import AbstractActionModule


class Search(AbstractActionModule):
    def perform(self, connector, reevaluate=False):
        # We do nothing here, the head is searching.
        connector.blackboard.schedule_ball_tracking()
        self.pop()


class StopAndSearch(Search):
    def perform(self, connector, reevaluate=False):
        connector.walking.stop_walking()
        super(StopAndSearch, self).perform(connector, reevaluate)