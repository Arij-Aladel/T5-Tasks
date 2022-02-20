import os
import json
import platform
import itertools
import numpy as np
import time
import string
import re
from collections import Counter

import datetime
import sys
from prettytable import PrettyTable

import argparse
from dataclasses import dataclass
import shutil
from pathlib import Path
from .utils import my_collate
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
# from transformers import T5ForConditionalGeneration
from transformers import (T5Tokenizer, 
                          set_seed, 
                          T5Config,
                          PreTrainedTokenizerBase,
                          SchedulerType,
                          AutoTokenizer,
                          get_scheduler,
                          BatchEncoding,
                          AutoConfig
                         )
import importlib
from .utils import get_cls_by_name, create_logger



@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        '''
        1. random_spans_noise_mask
        2. random span noise 
        3. create_sentinel_ids
        4. filter_input_ids
        if max length = 512 then len(example['input_ids']) = 568
        '''
        
        batch = BatchEncoding( # batch.keys() = input_ids
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )
         
        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        batch["input_ids"] = torch.tensor(self.filter_input_ids(input_ids, input_ids_sentinel))
        labels = self.filter_input_ids(input_ids, labels_sentinel)
        batch["labels"] = torch.tensor(labels)
        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )
        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]

    
    
def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length
    
    

class Tester:
    """
    This is a tester class for testing T5Mem Model

    Attributes:
    ...

    """

    def __init__(self, config: T5Config, args: argparse.ArgumentParser):
        self.config = config
        self.args = args
        self.experiment_path = Path(self.args.experiment_path)
        if not self.experiment_path.exists():
            Path(self.experiment_path).mkdir(parents=True)
        self.logger = create_logger(os.path.join(self.experiment_path, "log.txt"))

    def _create_tokenizer(self) -> T5Tokenizer:
        self.tokenizer =  AutoTokenizer.from_pretrained(self.args.tokenizer_path, config=self.config)    
#         from_pretrained('tokenizer/configs/SentencePieceUnigramTokenizer_wiki/32000', config=AutoConfig.from_pretrained('configs')
        self.config.vocab_size=len(self.tokenizer)    

    def _create_dataloders(self) -> tuple:

        """
        - a test dataloader that generates batches of src and tgt data
        """
        self.logger.info(f"Build test dataloader ...")
        per_worker_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
        test_dataset = HotpotDataset(self.args.data_path,
                            self.tokenizer,
                            self.config.max_source_len * self.config.num_input_sent, # Source_length 100
                            self.config.max_target_len, # targetlength   100
                            type= "dev",)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=per_worker_batch_size, #elf.args.batch_size,
            num_workers=0,  # num
            collate_fn=my_collate,
            worker_init_fn=np.random.seed(42),
            pin_memory=True,
        )


        return test_dataloader

    def _build_model(self) -> None:
        """
        Build model and update its configuration.
        """
        model_class = get_cls_by_name(self.args.model_name)
        self.model = model_class(self.config) 
        self.model.config = self.config
        # Additional step to guarantee that there is no conflict between vaocab size of the tokenizer and model
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.cuda()
        print("the model is:...................................................................")
        print(self.model)
        print("................................................................................")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build optimizer to be used in training.
        """
        # https://github.com/huggingface/transformers/blob/69511cdcaec8c1c7f0d7f378964eca0ce74ed5a8/src/transformers/trainer.py
        # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
        optimizer_cls = Adafactor if self.args.optimizer=='Adafactor' else AdamW
        if self.args.optimizer=='Adafactor':
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (0.9, 0.999),
                "eps": 0.000001
            }
        optimizer_kwargs["lr"] = self.args.lr
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        return optimizer

    def get_actual_seq_len(self, tokens):
        
        if '<pad>' in tokens:
            print("tokens from get_actual_seq_len:  ", tokens)
            return tokens[1:].index('<pad>')
        return len(tokens)

    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
                decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


    def test(   # TODO:
        self, test_dataloader: DataLoader) -> tuple:
        """
        Evaluate parallel test dataset, calculate evaluation loss, write hypothesis to file, and displays bleu score.
        """
        self.logger.info(f'start testing..')
        self.model.eval()
        if self.args.task =="mlm":
#             print("------------------  MLM  -------------------------")
            answer_dict = []  # from generate
            answer_dict_forward = []  # from generate
            ground_dict = []
            
            with torch.no_grad():
                for batch in tqdm(test_dataloader):
#                     print("batch keys: ", batch.keys()) batch keys:  dict_keys(['input_ids', 'labels'])

                    batch_loss = 0 # every batch has just input_ids and labels
                    source_ids = batch['input_ids'].cuda()
                    target_ids = batch["labels"].cuda()

                    outputs = self.model(input_ids=source_ids,
                                         labels=target_ids,
                                         output_hidden_states=True,
                                         output_attentions=True,
                                         return_dict=True, )


#                     print("outputs keys: ", outputs.keys())
# outputs keys:  odict_keys(['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions'])

                    mlm_answer = self.model.generate(source_ids)  
                    decoded_mlm_answer = [
                        self.tokenizer.decode(
                                  answer, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )
                            for answer in mlm_answer
                        ]
                    print("1: decoded_mlm_answer using self.model.generate(source_ids) -->> ")
                    print(decoded_mlm_answer)
                    print("********************************************************")

                    mlm_answer = self.model.generate(input_ids=source_ids,  # Is not used in metrics
                                                        eos_token_id=self.tokenizer.eos_token_id,
                                                        do_sample=True,
                                                        early_stopping=True, top_p=0.9, top_k=30,
                                                        length_penalty=1.5,
                                                        temperature=0.9,
                                                        repetition_penalty=2.0)    # generated from the model answer
#                     print("hotpot_answer keys: ", hotpot_answer.keys())
                    decoded_mlm_answer = [
                        self.tokenizer.decode(
                                  answer, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )
                            for answer in mlm_answer
                        ]
                    print("2: decoded_mlm_answer using self.model.generate(input_ids=source_ids,  eos_token_id=self.tokenizer.eos_token_id, do_sample=True, early_stopping=True, top_p=0.9, top_k=30,length_penalty=1.5,temperature=0.9, repetition_penalty=2.0) -->> ")
                    print(decoded_mlm_answer)
                    print("********************************************************")
                
                    mlm_answer = self.model.generate(input_ids=source_ids,  # Is not used in metrics
                                                        eos_token_id=self.tokenizer.eos_token_id,
                                                        do_sample=True,
                                                        early_stopping=True, top_k=40,
                                                        length_penalty=2.0,
                                                        temperature=0.5,
                                                        repetition_penalty=2.0)    # generated from the model answer
#                     print("hotpot_answer keys: ", hotpot_answer.keys())
                    decoded_mlm_answer = [
                        self.tokenizer.decode(
                                  answer, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )
                            for answer in mlm_answer
                        ]
                    print("3: decoded_mlm_answer using self.model.generate(input_ids=source_ids, eos_token_id=self.tokenizer.eos_token_id, do_sample=True, early_stopping=True, top_k=40, length_penalty=2.0, emperature=0.5, repetition_penalty=2.0)-->> ")
                    print(decoded_mlm_answer)
                    print("********************************************************")

                    ###################################### get ground truth  #####################################
                    target_ids[target_ids== -100] = 0  # reference answers
                    decoded_ground = [  # references
                        self.tokenizer.decode(
                            target, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        for target in target_ids
                    ]
                    
                    ground_dict += decoded_ground   # references
                
                    answer_dict += decoded_mlm_answer

                    print("*************************************************************************")
                    print("answer_dict resulted from model using generate method on valid set")
                    for key, value in enumerate(answer_dict):
                        print(key, ": ", value)
                    print("*************************************************************************")
                    print("ground_dict:")
                    for key, value in enumerate(ground_dict):
                        print(key, ": ", value)
            
#                     print(ground_dict)
                    print("*************************************************************************")


                    sentences = torch.cat([torch.stack(torch.chunk(sents, self.config.num_input_sent)) for sents in source_ids])
                    sents = [[self.tokenizer.convert_ids_to_tokens(t_id) for t_id in
                              sent.tolist()] for sent in sentences]
                    sents = [tokens[:self.get_actual_seq_len(tokens)] for tokens in sents]
                    max_logits = torch.argmax(outputs['logits'], dim=-1) # max_logits shape:  torch.Size([20, 40]) batch_size, target_length
                    model_output = [self.tokenizer.convert_ids_to_tokens(t_id) for t_id in
                              max_logits[0].tolist()]
                    decoded_model_output = [[self.tokenizer.convert_ids_to_tokens(t_id) for t_id in  # output of the mosel resulted from using forward
                              max_logit.tolist()] for max_logit in max_logits]
    
                    decoded_logits = [self.tokenizer.decode(
                                  answer, skip_special_tokens=True, clean_up_tokenization_spaces=True)                           
                            for answer in max_logits]
                    answer_dict_forward += decoded_logits
                
                    print("*************************************************************************")
                    print("answer_dict_forward: answers resulted using model forward ")
                    for key, value in enumerate(answer_dict_forward):
                        print(key, ": ", value)
                    print("*****************************Mode forward, reference tokens*************************************")
                    
#                     [self.tokenizer.convert_ids_to_tokens(t_id).replace('▁', '') for t_id in max_logits[0].tolist()]
                    # The real output that is supposed to be generated from the model
                    y_tokens = [[self.tokenizer.convert_ids_to_tokens(t_id) for t_id in
                                target_id.tolist()]for target_id in target_ids ]
        
                    decoder_input_tokens = [[self.tokenizer.convert_ids_to_tokens(t_id) for t_id in
                                self._shift_right(target_id).tolist()]for target_id in target_ids ]
#                     decoder_input_tokens = [tokens[:self.get_actual_seq_len(tokens)] for tokens in ground_selector_tokens]
        
#                     print("2: decoded_model_output[0]: ", decoded_model_output[0])
#                     print("2: ground_selector_tokens[0]: ", ground_selector_tokens[0])
                    for x,y in zip(decoded_model_output[0], y_tokens[0]):
                        print("decoded_model_output, Refrence tokens: ", "|  ",x,"|",y,"  |")

                    queries = sents    # in the encoder
                    model_output = model_output[:self.get_actual_seq_len(model_output)]
                    print("************************************************************************************************")
                    keys = sents   # in the encoder

                    return outputs['encoder_attentions'], outputs['cross_attentions'], outputs['decoder_attentions'], keys, decoder_input_tokens, decoder_input_tokens, queries, decoded_model_output


    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.init_checkpoint)
        self.model.load_state_dict(checkpoint["model_state"])
        best_bleu_score = checkpoint.get('best_bleu_score', 0)
        if 'amp' in checkpoint and self.args.fp16:
            self.amp.load_state_dict(checkpoint['amp'])
        init_iteration = checkpoint.get('epoch', 0)
        self.logger.info(f'Model was loaded from: {self.args.init_checkpoint}')


    def return_weights(self) -> None:
        """
        Train model on parallel dataset, evaluate dev data and save best model according to bleu if
        specified.
        """
        self.args.working_dir = str(Path(self.args.working_dir).expanduser().absolute())
        os.chdir(self.args.working_dir)
#         print("os.path.join(self.args.working_dir,os.path.basename(__file__): ", os.path.join(self.args.working_dir,os.path.basename(__file__)))
        shutil.copyfile(os.path.join(self.args.working_dir+"/src",os.path.basename(__file__)), os.path.join(self.experiment_path, os.path.basename(__file__)))
        

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])



        self.logger.info(f'FP16: {self.args.fp16}')
        args_dict = dict(vars(self.args))
        args_dict['ENV'] = {}
        for env_var in ['CUDA_VISIBLE_DEVICES']: 
            args_dict['ENV'][env_var] = os.environ.get(env_var, '')

        args_dict['MACHINE'] = platform.node()
        args_dict['sub_batch_size'] = self.args.batch_size
        batch_size = args_dict['batch_size'] * self.args.gradient_accumulation_steps * len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        batch_string = str(args_dict['batch_size']) + "_" + str(self.args.gradient_accumulation_steps) + "_" + str(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        args_dict['batch_size'] = batch_size
        args_dict["lr_scheduler_type"]="<SchedulerType.LINEAR: 'linear'>"
        file_path = 'args_config_'+str(self.args.batch_size)+"_"+str(self.args.lr)+'.json'
        json.dump(args_dict, open(self.experiment_path / file_path, 'w'), indent=4)
        log_dir = self.args.experiment_path+'runs_'+batch_string+"_"+str(self.args.lr)
        


        self.logger.info("Building tokenizer")
        self._create_tokenizer()
        self.logger.info(self.tokenizer)
        

        if self.args.max_seq_length is None:
            self.max_seq_length = self.tokenizer.model_max_length
            if self.max_seq_length > 512:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
                )
                self.max_seq_length = 512
        else:
            if self.args.max_seq_length > self.tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({self.args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({self.tokenizer.model_max_length}). Using self.max_seq_length={self.tokenizer.model_max_length}."
                )
            self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)



        self.logger.info(f"Experiment Output Path: \n {self.experiment_path}")
        self.logger.info("Building model...")
        # 6. building the model
        self._build_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        file_path = 'config_model.json'
        self.model.config.to_json_file(self.experiment_path / file_path)



        self.logger.info(f"Training will be done with this configuration: \n {self.model.config}\n")
        self.logger.info("model:\n")
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        self.logger.info(table)
        self.logger.info(f"Total Trainable Params: {total_params}\n")
        self.logger.info(f'{self.model}\n')

        if self.args.init_checkpoint:
             self._load_checkpoint() 
                
        if self.args.task == "mlm":
            self.expanded_inputs_length, self.targets_length = compute_input_and_target_lengths(
                inputs_length=self.max_seq_length,
                noise_density=self.args.mlm_probability,
                mean_noise_span_length=self.args.mean_noise_span_length,
            )
            
            self.collate = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.args.mlm_probability,
            mean_noise_span_length=self.args.mean_noise_span_length,
            input_length=self.max_seq_length,
            target_length=self.targets_length,
            pad_token_id=self.config.pad_token_id,
            decoder_start_token_id = self.model.config.decoder_start_token_id
        )
            test_dataset = torch.load(self.args.test_file_path)
            test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=0,
            collate_fn=self.collate,
            worker_init_fn=np.random.seed(42),
            pin_memory=True,
        )            
            encoder_attentions, cross_attentions, decoder_attentions, keys, ground_selector_tokens, decoder_input_tokens, queries, decoded_model_output = self.test(test_dataloader)
           ####### encoder_attentions, cross_attentions, decoder_attentions , selector_attention, tokens, y_tokens, sents = main()
            return encoder_attentions, cross_attentions, decoder_attentions , keys, ground_selector_tokens, ground_selector_tokens, queries, decoded_model_output
            
