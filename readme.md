# RL-PROMPT-Injection
The core idea for this repo came from *RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning*. [1]

This library aims to identify prompts that trigger the language model to generate text that scores highly with the evaluator model. These prompts become red-team prompts when the evaluator is a classification model that identifies toxicity or sentiment. 

[1] https://arxiv.org/abs/2205.12548

## 🚨 Health Warning 🚨
This library objective is to train RL agents to generate prompts that score highly with the evaluation model. In the main example, developing prompts to trigger toxic text generation you may see text that contains toxic/unsavoury content. Please tread carefully.
