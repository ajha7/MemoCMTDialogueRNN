from configs.hubert_base import Config as BaseConfig

class Config(BaseConfig):
    def add_args(self, **kwargs):
        super().add_args(**kwargs)
        self.model_type = "MemoCMTDialogueRNN"
        self.trainer = "DialogueTrainer"

        self.conversation_aware = True
        self.num_speakers = 2
        self.fusion_head_output_type = "cls"
        self.dialogue_hidden_size = 128
        # Strict: turn t is predicted from turns t-w+1..t only. 1 = no context, None = whole dialogue.
        self.context_window = 6
        self.freeze_feature_extractor = False
        self.text_unfreeze = True
        self.audio_unfreeze = False

        self.batch_size = 2
        self.memo_chunk_size = 16
        self.use_gradient_checkpointing = True
        self.use_amp = True
        self.efficient_attention = True
        self.max_grad_norm = 0.5
        self.learning_rate = 5e-6
        self.dropout = 0.2
        self.fusion_learning_rate = 3e-5
        self.dialogue_learning_rate = 1e-4
        self.num_epochs = 25
        self.learning_rate_gamma = 0.1
        self.learning_rate_step_size = 30
        self.ablate_audio = True

        self.fast_first_epoch = True
        self.skip_first_epoch_eval = True
        self.log_every_n_steps = 20

        for k, v in kwargs.items():
            setattr(self, k, v)