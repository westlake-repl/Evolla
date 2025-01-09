import torch
from torch import nn
from torch.nn import functional as F
from transformers import EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer

from .encoder_interface import register_encoder
from .fusion_module import SequenceCompressorResampler


@register_encoder
class SaProtSequenceEncoder(nn.Module):
    def __init__(
        self,
        config_path: str,
        load_pretrained: bool = True,
        fusion_module: dict = None,
        **kwargs,
    ):
        super().__init__()
        if load_pretrained:
            # self.model = EsmModel.from_pretrained(config_path)
            self.model = EsmForMaskedLM.from_pretrained(config_path)
            self.config = EsmConfig.from_pretrained(config_path)
        else:
            self.config = EsmConfig.from_pretrained(config_path)
            # self.model = EsmModel(self.config)
            self.model = EsmForMaskedLM(self.config)

        self.tokenizer = EsmTokenizer.from_pretrained(config_path)

        fusion_cls = fusion_module.pop("cls", None)
        if fusion_cls is None or fusion_cls == "SequenceCompressorResampler":
            self.resampler = SequenceCompressorResampler(**fusion_module)
        else:
            raise ValueError(f"Unknown fusion module class: {fusion_cls}")

    @property
    def num_layers(self):
        return len(self.model.encoder.layer)

    def sequence_encode(self, seqs):
        """
        Encode protein sequence into protein representation
        """
        seqs = [seq if seq is not None else "" for seq in seqs]
        protein_tokens = self.tokenizer.batch_encode_plus(
            seqs, return_tensors="pt", truncation=True, max_length=1026, padding=True
        ).to(self.model.device)

        protein_output = self.model(
            protein_tokens["input_ids"],
            protein_tokens["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )

        protein_embeds = protein_output.hidden_states[-1]

        mask = protein_tokens["attention_mask"]

        return protein_embeds, mask

    def forward(self, seqs):
        # create batch mask for seqs
        seqs_batch_mask = torch.tensor(
            [True if seq is not None else False for seq in seqs]
        )
        # print("this is structure encoder", flush=True)
        sequence_embeds, mask = self.sequence_encode(seqs)

        sequence_repr = self.resampler(sequence_embeds, mask)

        return sequence_repr, sequence_embeds, mask, seqs_batch_mask

