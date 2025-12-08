from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.text_unfreeze_layers = [-1, -2]
        self.text_unfreeze = True
        self.optimizer_type = "AdamW"
        self.dropout_rate = 0.2
        self.max_grad_norm = 0.5
        self.learning_rate = 5e-06
        self.use_amp = True
        self.prefetch_factor = 2
        self.persistent_workers = True
        self.pin_memory = True
        self.num_workers = 4
        self.batch_size = 2
        self.num_epochs = 20
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.adam_eps = 1e-08
        self.adam_weight_decay = 0
        self.sdg_weight_decay = 1e-06
        self.save_freq = 100000
        self.max_to_keep = 1
        self.save_best_val = True
        self.save_all_states = False
        self.linear_layer_output = [128]
        self.linear_layer_last_dim = 64
        self.num_attention_head = 8

        self.loss_type = "CrossEntropyLoss"

        self.checkpoint_dir = "/content/drive/MyDrive/MemoCMTDialogueRNN/MemoCMT/src/checkpoints/IEMOCAP/MemoCMT_bert_hubert_base/20251207-163915"

        self.model_type = "MemoCMT"

        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = True

        self.audio_encoder_type = "hubert_base"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.fusion_dim: int = 768

        # Dataset
        self.data_name: str = "IEMOCAP"
        self.data_root = "../src/data/IEMOCAP"
        self.data_valid: str = "val.pkl"
        # self.text_max_length: int = 297
        # self.audio_max_length: int = 128000  # 160220
        self.text_max_length = 128
        self.audio_max_length = 64000

        # Config name
        self.name = (
            f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}"
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
