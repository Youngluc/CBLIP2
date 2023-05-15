"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ChineseBLIP2.common.registry import registry
from ChineseBLIP2.models.blip2_models.blip2 import Blip2Base, disabled_train
from ChineseBLIP2.models.blip2_models.modeling_gpt import GPT2LMHeadModel
from transformers import AutoTokenizer, GPT2Tokenizer


@registry.register_model("blip2_gpt")
class Blip2GPT(Blip2Base):
    """
    BLIP2 gpt model.
    Supported model types:
        - pretrained_gpt3.5b: pretrained model with IDEA-wenzhong-chineseGPT3.5b
    Usage:
        >>> from ChineseBLIP2.models import load_model
        >>> model = load_model("blip2_gpt", "pretrained_gpt3.5b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_gpt3.5b": "configs/models/blip2/blip2_pretrain_gpt3.5b.yaml",
    }

    def __init__(
        self,
        vit_model="chinese_clip_H",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        gpt_model="IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese",
        prompt="",
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        if 'chinese' in vit_model:
            self.Qformer, self.query_tokens = self.init_Qformer_Chinese(
                num_query_token, self.visual_encoder.num_features
            )
        else:
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model, use_fast=False)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(
            gpt_model, torch_dtype=torch.float16
        )
        for name, param in self.gpt_model.named_parameters():
            param.requires_grad = False
            
        self.eos_token_id = self.gpt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        # missing pad token
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token

        self.gpt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.gpt_model.transformer.embed_dim
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.gpt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_gpt = self.gpt_proj(query_output.last_hidden_state)
        atts_gpt = torch.ones(inputs_gpt.size()[:-1], dtype=torch.long).to(image.device)

        self.gpt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        gpt_tokens = self.gpt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = gpt_tokens.input_ids.masked_fill(
            gpt_tokens.input_ids == self.gpt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_gpt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.gpt_model.transformer.wte(gpt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_gpt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_gpt, gpt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.gpt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_gpt = self.gpt_proj(query_output.last_hidden_state)
            atts_gpt = torch.ones(inputs_gpt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            gpt_tokens = self.gpt_tokenizer(prompt, return_tensors="pt").to(
                image.device
            )
            input_ids = gpt_tokens.input_ids
            attention_mask = torch.cat([atts_gpt, gpt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_gpt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_gpt.repeat_interleave(num_beams, dim=0)

            outputs = self.gpt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = gpt_tokens.input_ids.shape[1]
            output_text = self.gpt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        gpt_model = cfg.get("gpt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            gpt_model=gpt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
