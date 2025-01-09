import pytorch_lightning as pl
from model.model_interface import register_model
from utils.easydict import MyEasyDict
import torch

from .encoder_interface import EncoderInterface
from .llm_interface import LLMInterface

@register_model
class EvollaModel(pl.LightningModule):
    def __init__(self,
        config: MyEasyDict,
        **kwargs):
        """
        Initialize the Evolla.
        Args:
            config (MyEasyDict): Configuration of the Evolla.
        """
        super().__init__()
        self.verbose = config.get('verbose', False)
        self.config = config
        self.generate_config = kwargs.pop('generate_config', {})

        if len(self.generate_config) == 0:
            print("Warning: No generate config is provided, the generate config now is \{\}")
        else:
            print("Generate config is provided, the generate config is: ", self.generate_config)
        
        self.initialize_model()

        self.special_pad_id = -100

    @staticmethod
    def init_protein_encoder(config: dict):
        """
        Initialize protein encoder
        Args:
            config: A dictionary containing the configuration of the protein encoder

        Returns:
            A protein encoder
        """
        encoder_py_path = config.pop("cls")
        model = EncoderInterface.init_encoder(encoder_py_path, **config)
        return model

    @staticmethod
    def init_structure_encoder(config: dict):
        """
        Initialize structure encoder
        Args:
            config: A dictionary containing the configuration of the structure encoder
        Returns:
            A structure encoder
        """
        encoder_py_path = config.pop("cls")
        model = EncoderInterface.init_encoder(encoder_py_path, **config)
        return model

    @staticmethod
    def init_msa_transformer_encoder(config: dict):
        """
        Initialize protein encoder
        Args:
            config: A dictionary containing the configuration of the protein encoder

        Returns:
            A protein evoformer encoder
        """
        msa_transformer_py_path = config.pop("cls")
        model = EncoderInterface.init_encoder(msa_transformer_py_path, **config)
        return model

    @staticmethod
    def init_llm(config: dict):
        """
        Initialize LLM
        Args:
            config: A dictionary containing the configuration of the LLM

        Returns:
            A LLM
        """
        llm_py_path = config.pop("cls")
        model = LLMInterface.init_llm(llm_py_path, **config)
        return model

    def initialize_model(self) -> None:
        """Initialize the Evolla model."""
        # torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        if "protein_encoder" in self.config:
            if self.verbose:
                print("Loading Sequence Encoder...", flush=True)
            self.protein_encoder = self.init_protein_encoder(
                self.config["protein_encoder"]
            )
        else:
            self.protein_encoder = None

        if "msa_encoder" in self.config:
            if self.verbose:
                print("Loading MSA Tranformer Encoder...", flush=True)
            self.msa_encoder = self.init_msa_transformer_encoder(
                self.config["msa_encoder"]
            )
        else:
            self.msa_encoder = None

        if "structure_encoder" in self.config:
            if self.verbose:
                print("Loading Structure Encoder...", flush=True)
            self.structure_encoder = self.init_structure_encoder(
                self.config["structure_encoder"]
            )
        else:
            self.structure_encoder = None
        # print("Loading Fusion Module...", flush=True)
        # self.fusion_module = self.init_fusion_module(self.config["fusion_module"])
        if self.verbose:
            print("Loading LLM...", flush=True)
        self.llm = self.init_llm(self.config["llm"])
        self.llm_tokenizer = self.llm.tokenizer

        if self.protein_encoder is not None:
            self.freeze_protein_encoder_layers()

        if self.structure_encoder is not None:
            self.freeze_structure_encoder_layers()

        if self.msa_encoder is not None:
            self.freeze_msa_encoder_layers()

        self.freeze_llm_layers()

    def freeze_protein_encoder_layers(self):
        for name, param in self.protein_encoder.named_parameters():
            param.requires_grad = False
            if "resampler" in name:
                param.requires_grad = True

    def freeze_structure_encoder_layers(self):
        for name, param in self.structure_encoder.named_parameters():
            param.requires_grad = False
            if "resampler" in name:
                param.requires_grad = True

    def freeze_msa_encoder_layers(self):
        for name, param in self.msa_encoder.named_parameters():
            param.requires_grad = False
            if "resampler" in name:
                param.requires_grad = True

    def freeze_llm_layers(self):
        for name, param in self.llm.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def input_process(
        self,
        questions: list,
        answers: list = None,
        # raw_text_lists: list = None,
    ):
        """
        Args:
            protein_embeds: encoded embedding of protein sequence
            templates: template used as container of question and answer pair
            questions: A list of prompts.
            answers: A list of answers.
        """
        return self.llm.input_process(
            questions=questions,
            answers=answers,
            max_length=self.config["text_length"],
            special_pad_id=self.special_pad_id,
            )

    def forward(
        self,
        seqs: tuple,
        foldseeks: tuple,
        questions: list,
        answers: list,
        msa_embeds: torch.Tensor = None,
        msa_atts: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass of the Evolla model.
        Args:
            seqs (tuple): Amino acid sequences of proteins.
            foldseeks (tuple): Foldseek sequences of proteins.
            questions (list): A list of prompts.
            answers (list): A list of answers.
            msa_embeds (torch.Tensor, Optional): MSA embeddings.
            msa_atts (torch.Tensor, Optional): MSA attention masks.
        
        Returns:
            return_dict (dict): A dictionary containing the predicted logits, prompts, answers, and raw text masks.
            labels (torch.Tensor): A tensor containing the labels.
        """

        if self.protein_encoder is not None:
            resampler_protein_repr, protein_repr, protein_attn, protein_batch_mask = self.protein_encoder(seqs)
        else:
            resampler_protein_repr = None
            protein_batch_mask = None
            protein_repr = None
            protein_attn = None

        if self.structure_encoder is not None:
            resampler_structure_repr, structure_repr, structure_attn, structure_batch_mask = self.structure_encoder(foldseeks)
        else:
            resampler_structure_repr = None
            structure_batch_mask = None
            structure_repr = None
            structure_attn = None

        if self.msa_encoder is not None:
            resampler_msa_repr, msa_repr, msa_attn, msa_batch_mask = self.msa_encoder(
                msa_embeds,
                msa_atts,
            )
        else:
            resampler_msa_repr = None
            msa_repr = None
            msa_attn = None
            msa_batch_mask = None

        input_ids, embeds, attns, labels, raw_text_masks = self.input_process(
            # protein_embeds=resampler_protein_repr,
            # structure_embeds=resampler_structure_repr,
            # templates=templates,
            questions=questions,
            answers=answers,
            # raw_text_lists=raw_text_lists,
        )

        outputs = self.llm.forward(
            input_ids=input_ids,
            inputs_embeds=embeds,
            inputs_mask=attns,
            protein_feats=resampler_protein_repr,
            structure_feats=resampler_structure_repr,
            msa_feats=resampler_msa_repr,
            protein_batch_mask=protein_batch_mask,
            structure_batch_mask=structure_batch_mask,
            msa_batch_mask=msa_batch_mask,
        )
        logits = outputs.logits
        
        return_dict = {
            "logits": logits,
            "prompts": questions,
            "answers": answers,
            "raw_text_masks": raw_text_masks,
        }
        if "comment_types" in kwargs:
            return_dict["comment_types"] = kwargs["comment_types"]

        return return_dict, labels


    def generate(
        self,
        seqs: tuple,
        foldseeks: tuple,
        questions: list,
        msa_embeds: torch.Tensor = None,
        msa_atts: torch.Tensor = None,
        **kwargs,
    ) -> str:
        """
        Generate answer for the question.
        Args:
            seqs (tuple): Amino acid sequences of proteins.
            foldseeks (tuple): Foldseek sequences of proteins.
            questions (list): A list of questions.
            msa_embeds (torch.Tensor, Optional): MSA embeddings.
            msa_atts (torch.Tensor, Optional): MSA attention masks.

        Returns:
            answers (list): A list of predicted answers.
        """

        with torch.no_grad():
            if self.protein_encoder is not None:
                (
                    resampler_protein_repr,
                    protein_repr,
                    protein_attn,
                    protein_batch_mask,
                ) = self.protein_encoder(seqs)
            else:
                resampler_protein_repr = None
                protein_batch_mask = None
                protein_repr = None
                protein_attn = None

            if self.structure_encoder is not None:
                (
                    resampler_structure_repr,
                    structure_repr,
                    structure_attn,
                    structure_batch_mask,
                ) = self.structure_encoder(foldseeks)
            else:
                resampler_structure_repr = None
                structure_batch_mask = None
                structure_repr = None
                structure_attn = None

            if self.msa_encoder is not None:
                resampler_msa_repr, msa_repr, msa_attn, msa_batch_mask = self.msa_encoder(
                    msa_embeds,
                    msa_atts,
                )
            else:
                resampler_msa_repr = None
                msa_batch_mask = None
                msa_repr = None
                msa_attn = None

            input_ids, embeds, attns, labels, raw_text_masks = self.input_process(
                questions=questions,
            )

        predicted_answer = self.llm.generate(
            input_ids=input_ids,
            inputs_mask=attns,
            protein_feats=resampler_protein_repr,
            structure_feats=resampler_structure_repr,
            msa_feats=resampler_msa_repr,
            protein_batch_mask=protein_batch_mask,
            structure_batch_mask=structure_batch_mask,
            msa_batch_mask=msa_batch_mask,
            **kwargs,
        )

        return self.llm.tokenizer.batch_decode(
            predicted_answer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )