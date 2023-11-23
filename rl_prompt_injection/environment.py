from typing import Callable

import numpy as np
import pandas as pd
import torch
import gymnasium as gym


class ToxicityEnvironment(gym.Env):
    def __init__(
        self,
        llm: Callable[[str], str],
        reward_engine,
        state_engine,
        action_engine,
        instruction_prompt: str | None,
        texts: np.array,
        experiement_dir: str,
        eps_len: int = 10,
        log: bool = True,
        log_interval: int = 1000

    ):
        self.texts: np.array = texts
        self.eps_len: int = eps_len
        self.llm: Callable[[str], str] = llm
        self.reward_engine = reward_engine
        self.state_engine = state_engine
        self.action_engine = action_engine

        self.instruction_prompt = (
            instruction_prompt if instruction_prompt is not None else ""
        )

        self.observation_space = self.state_engine.observation_space
        self.action_space = self.action_engine.action_space
        self.step_counter: int = 0
        self.episode_counter: int = 1
        self.experiment_dir: str = experiement_dir
        self.setup()
        self.setup_log()
        self.do_logging = log
        self.log_interval:int = log_interval
    
    def setup_log(self):
        self._log: list = list()

    def log(self):
        df = pd.DataFrame(self._log)
        COLS_TO_EXPLODE = [
            "state_toxicity_scores",
            "response_toxicity_scores",
            "rewards"
        ]
        for col in COLS_TO_EXPLODE:
            length = len(df[col].iloc[0])
            for idx in range(length):
                df[f"{col}_{str(idx)}"] = df[col].apply(lambda l: l[idx])

        df.to_csv(f"{self.experiment_dir}/results_{self.episode_counter}.csv", index=False)
        self.setup_log()

    def setup(self):
        
        self.text_state: str = (
            self.instruction_prompt + " " + np.random.choice(self.texts)
        )

        self.state = self.state_engine.encode_state(self.text_state)

        self.prompt: str = list()

        self.rewards: list() = list()
        self.prev_state_text: str = self.text_state
        self.state_toxicity_scores: list = [self.reward_engine.model(self.text_state)]
        self.responses = list()
        self.response_toxicity_scores: list = list()
        self.actions: list = list()
        

    def _get_obs(self) -> torch.Tensor:
        return self.state

    def _get_info(self) -> dict:
        return {
            "base_state": self.text_state,
            "prompt": self.prompt,
            "state_toxicity_scores": self.state_toxicity_scores,
            "responses": self.responses,
            "response_toxicity_scores": self.response_toxicity_scores,
            "state": self.state,
            "rewards": self.rewards,
            "actions": self.actions
        }

    def reset(self, seed: int = None, options=None) -> tuple[torch.Tensor, dict]:
        self.step_counter = 0
        self.setup()
        return self._get_obs(), self._get_info()

    def step(self, action: str) -> str:
        self.actions.append(action)

        self.step_counter += 1
        if self.step_counter == self.eps_len:
            done = True
        else:
            done = False

        decoded_action = self.action_engine.decode_action(action)
        self.text_state += " " + decoded_action
        self.state = self.state_engine.encode_state(self.text_state)

        self.prompt.append(decoded_action)
        # self.action_toxicity: float = 0.0
        response = self.llm(self.text_state)

        reward, state_toxicity, response_toxicity = self.reward_engine.calculate_reward(
            self.text_state, response, done
        )

        self.prev_state = self.state
        self.state = self.state_engine.encode_state(self.text_state)
        self.prev_state_text = self.text_state
        self.state_toxicity_scores.append(state_toxicity)
        self.responses.append(response)
        self.response_toxicity_scores.append(response_toxicity)
        self.rewards.append(reward)

        info = self._get_info()

        if done:
            self.episode_counter += 1
            self._log.append(info)

            if self.do_logging and self.episode_counter%self.log_interval == 0:
                self.log()
            
            self.reset()

        return self.state, reward, done, False, info
