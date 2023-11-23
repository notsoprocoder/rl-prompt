
class ToxicityEnvironment(object):
    def __init__(
            self,
            llm: Callable[[str],str],
            reward_engine,
            state_engine,
            action_engine,
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

    def setup(self):
        self.state: str = np.random.choice(self.texts)
        self.encoded_state = self.state_engine.encode_state(self.state)
        self.state_toxicity: float = self.toxicity_model(self.state)
        self.action_toxicity: float = 0.0
        self.prompt: str = ""
        self.prompt_counter: int = 0

    def _get_obs():
        pass
    def _get_info():
        pass
    def reset():

    def step(self, action: str) -> str:
        if self.prompt_counter==self.eps_len:
            done=True

        mapped_action = self.action_space_engine.map_action(action)
        self.state: str += " " + mapped_action
        # self.action_toxicity: float = 0.0
        response = self.llm(self.state + " " + self.prompt)
        reward = self.reward_engine.calculate_reward(self.state, self.state_toxicity, response, self.mapped_action, done)

        self.encoded_state = self.state_engine.encode_state(self.state)

        self.prev_state = self.state
        self.state_toxicity = self.toxicity_model(self.state)
        return self.state, reward, False, info, done

        