
class ToxicityEnvironment(object):
    def __init__(
            self,
            llm: Callable[[str],str],
            reward_engine,
            state_engine,
            action_engine,
            instruction_prompt: str | None,
            texts: np.array,
            toxicity_model: Callable[[str], float],
            eps_len: int = 10,
            ):
        self.texts: np.array = texts
        self.toxicity_model: Callable[[str], float] = toxicity_model
        self.eps_len: int = eps_len
        self.llm: Callable[[str], str] = llm
        self.reward_engine = reward_engine
        self.state_engine = state_engine
        self.action_engine = action_engine

        self.instruction_prompt = instruction_prompt if instruction_prompt is not None else ""

        self.obs_space = self.state_engine.observation_space
        self.action_space = self.state_engine.observation_space

    def setup(self):
        self.text_state: str = self.instruction_prompt + " " + np.random.choice(self.texts)
        
        self.state = self.state_engine.encode_state(self.state)

        self.prompt: str = list()
        self.step_counter: int = 0
        
        self.prev_state_text = self.text_state
        self.state_toxicity_scores = [self.toxicity_model(self.state)]
        self.responses = list()
        self.response_toxicity_scores = list()

    def _get_obs(self) -> torch.Tensors:
        return self.state
    
    def _get_info(self) -> dict:
        return {
            "base_state": self.text_state,
            "prompt": self.prompt,
            "state_toxicity_scores": self.state_toxicity_scores,
            "responses": self.responses,
            "response_toxicity_scores": self.response_toxicity_scores,
            "state": self.state
        }
    
    def reset(self, seed: int=None, options=None) -> tuple[torch.Tensor, dict]:
        self.setup()
        return self._get_obs(), self._get_info()

    def step(self, action: str) -> str:
        if self.step_counter==self.eps_len:
            done=True
        else: 
            self.step_counter += 1

        decoded_action = self.action_space_engine.decode_action(action)
        self.text_state += " " + decoded_action
        self.prompt.append()
        # self.action_toxicity: float = 0.0
        response = self.llm(self.text_state)

        reward, response_toxicity, state_toxicity = self.reward_engine.calculate_reward(self.text_state, response, self.mapped_action, done)

        self.encoded_state = self.state_engine.encode_state(self.state)

        self.prev_state = self.state
        self.state = self.state_engine.encode_state(self.state_text)
        self.prev_state_text = self.state_text
        self.state_toxicity_scores.append(state_toxicity)
        self.responses.append(response)
        self.response_toxicity_scores.append(response_toxicity)
        

        return self.state, reward, False, self._get_info(), done

        