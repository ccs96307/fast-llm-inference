from functools import wraps
import time
import threading
import subprocess

import torch


def log_gpu_status_decorator(log_file_prefix="gpu_log", interval=5, gpu_ids=None):
    """
    Decorator to log GPU hardware status periodically during function execution,
    allowing selection of specific GPUs.

    :param log_file_prefix: Prefix for the log file name. The final name will include a timestamp.
    :param interval: Interval (in seconds) for logging GPU stats.
    :param gpu_ids: List of GPU IDs to log. If None, logs all available GPUs.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            log_file_path = f"{log_file_prefix}_{timestamp}.csv"

            stop_event = threading.Event()
            cumulative_power = {}

            def log_gpu_status():
                """
                Logs GPU hardware stats and calculates cumulative power consumption.
                """
                with open(log_file_path, "w") as log_file:
                    log_file.write("timestamp,gpu_id,temperature,power_draw,utilization_gpu,memory_used,cumulative_power\n")
                try:
                    while not stop_event.is_set():
                        log_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        
                        # Run nvidia-smi to fetch GPU stats
                        result = subprocess.run(
                            ["nvidia-smi", 
                             "--query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used", 
                             "--format=csv,noheader,nounits"],
                            stdout=subprocess.PIPE, text=True
                        )
                        
                        gpu_info = result.stdout.strip()
                        if gpu_info:
                            for line in gpu_info.splitlines():
                                fields = line.split(", ")
                                gpu_id = int(fields[0])

                                # Skip GPUs not in the specified list
                                if gpu_ids is not None and gpu_id not in gpu_ids:
                                    continue

                                power_draw = float(fields[2])  # Current power draw in watts

                                # Update cumulative power consumption
                                if gpu_id not in cumulative_power:
                                    cumulative_power[gpu_id] = 0.0
                                cumulative_power[gpu_id] += power_draw * (interval / 3600)  # kWh formula

                                # Log data with cumulative power
                                log_entry = f"{log_timestamp},{line},{cumulative_power[gpu_id]:.6f}\n"
                                with open(log_file_path, "a") as log_file:
                                    log_file.write(log_entry)

                        time.sleep(interval)
                except Exception as e:
                    print(f"Error during GPU logging: {e}")

            log_thread = threading.Thread(target=log_gpu_status, daemon=True)
            log_thread.start()
            
            try:
                return func(*args, **kwargs)
            finally:
                stop_event.set()
                log_thread.join()
                print(f"GPU logging stopped. Logs saved to: {log_file_path}")

        return wrapper
    return decorator



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
        target_matchness: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        max_step_draft: int,
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
