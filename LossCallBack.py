from transformers import TrainerCallback


class LossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.test_losses = []

    def on_step_end(self, args, state, control, model, **kwargs):
        if len(state.log_history) != 0 and 'loss' in state.log_history[-1]:
            self.losses.append(state.log_history[-1]['loss'])
        if len(state.log_history) != 0 and 'eval_loss' in state.log_history[-1]:
            self.test_losses.append(state.log_history[-1]['eval_loss'])
            self.losses.append(state.log_history[-2]['loss'])
