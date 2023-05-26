from typing import Optional

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from ..base import Actor


class AutoActor(Actor):
    """
    GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LoRa layer.
        lora_train_bias (str): Bias training strategy for the LoRa layer.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[AutoConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = AutoModelForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = AutoModel(config)
        else:
            model = AutoModel(AutoConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)
