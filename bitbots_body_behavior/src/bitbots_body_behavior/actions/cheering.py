import rospy
from dynamic_stack_decider.abstract_action_element import AbstractActionElement


class Cheering(AbstractActionElement):
    def __init__(self, blackboard, dsd, parameters=None):
        super(Cheering, self).__init__(blackboard, dsd, parameters)
        self.animation = blackboard.animations["cheering"]
        # needed so we see if we already started the animation
        self.first = True

    def perform(self, reevaluate=False):
        if not self.blackboard.animation.is_animation_busy() and self.first:
            self.blackboard.animation.play_animation(self.animation)
            self.first = False
        elif not self.blackboard.animation.is_animation_busy():
            self.pop()
