import torch


class AdaptiveDraftExitAduster:
    def __init__(
        self,
        target_matchness: float = 0.9,
        beta1: float = 0.5,
        beta2: float = 0.9,
        epsilon: float = 0.01,
        max_step_draft: int = 8,
    ):
        """Initialize DraftExitingAdjuster parameters
        :param target_matchness: matching degree target value
        :param beta1: sliding average coefficient for matching degree update
        :param beta2: th_stop_draft smooth update coefficient
        :param epsilon: Adjust the step size each time
        :param max_step_draft: The maximum number of steps for draft generation
        """

        self.save_init_status(
            target_matchness=target_matchness,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_step_draft=max_step_draft,
        )
        self.reset()

    def save_init_status(
        self,
        target_matchness: float = 0.9,
        beta1: float = 0.5,
        beta2: float = 0.9,
        epsilon: float = 0.01,
        max_step_draft: int = 8,
    ) -> None:
        self.init_target_matchness = target_matchness
        self.init_beta1 = beta1
        self.init_beta2 = beta2
        self.init_epsilon = epsilon
        self.init_max_step_draft = max_step_draft

    def reset(self) -> None:
        self.target_matchness = self.init_target_matchness
        self.beta1 = self.init_beta1
        self.beta2 = self.init_beta2
        self.epsilon = self.init_epsilon
        self.max_step_draft = self.init_max_step_draft

        # Dynamic status
        self.curr_matchness = 0.0
        self.stop_draft_threshold = 0.5
        self.step_num = 0

    def update(self, num_matched_tokens, num_drafted_tokens) -> None:
        # Update matchness
        matchness = num_matched_tokens / num_drafted_tokens
        self.curr_matchness = self.beta1 * self.curr_matchness + (1 - self.beta1) * matchness

        # Calculate new exit threshold
        if num_drafted_tokens == self.max_step_draft:
            new_stop_draft_threshold = self.stop_draft_threshold

        elif self.curr_matchness <= self.target_matchness:
            new_stop_draft_threshold = self.stop_draft_threshold + self.epsilon
        else:
            new_stop_draft_threshold = self.stop_draft_threshold - self.epsilon

        self.stop_draft_threshold = self.beta2 * self.stop_draft_threshold + (1 - self.beta2) * new_stop_draft_threshold
        self.step_num += 1

    def should_exit(self, draft_prob: torch.FloatTensor) -> bool:
        return draft_prob < self.stop_draft_threshold

    def get_state(self):
        return {
            "curr_matchness": self.curr_matchness,
            "stop_draft_threshold": self.stop_draft_threshold,
            "step_num": self.step_num
        }


def calculate_continuous_acceptance(acceptance_mask: torch.BoolTensor) -> int:
    continuous_acceptance = 0
    for accepted in acceptance_mask.long().squeeze(0):
        if accepted == 1:
            continuous_acceptance += 1
        else:
            break
    return continuous_acceptance
