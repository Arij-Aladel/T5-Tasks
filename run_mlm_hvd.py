import os
import re
import csv
import argparse
import json
import platform
import itertools
import importlib
import random
import math
import traceback
import shutil

import time
import datetime
from pathlib import Path
import horovod.torch as hvd
from datasets import load_dataset

import numpy as np
from itertools import chain

from prettytable import PrettyTable
from einops import rearrange
from tqdm import tqdm
from typing import Dict, List, Optional

from src.utils import  EarlyStopping
from src.utils import create_logger
from src.utils import get_cls_by_name
from src.utils import load_config

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers.optimization import Adafactor, AdamW, AdafactorSchedule
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



def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)  #  # if you are using multi-GPU.
    set_seed(seed)



seed_everything(42)

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
    
    


class Trainer:
    """
    This is a trainer class for training T5MemForConditionalGeneration model.

    Attributes:
        config: model configuration
        args :  user variable parameters to train the model
    """

    def __init__(self, config: T5Config, args: argparse.ArgumentParser):
        self.config = config
#         conf_dict = config.get_config_dict(args.config_path)
        self.args = args
        self.experiment_path = Path(self.args.experiment_path)
        if not self.experiment_path.exists():
            Path(self.experiment_path).mkdir(parents=True)
        self.logger = create_logger(os.path.join(self.experiment_path, "log.txt"))

    def _create_tokenizer(self) -> T5Tokenizer:
        print("*************  self.args.tokenizer_path  *******************")
        print(self.args.tokenizer_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)  
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, config=self.config)    

        self.config.vocab_size=len(self.tokenizer)            
        if self.args.max_seq_length is None:
            self.max_seq_length = tokenizer.model_max_length
            if self.max_seq_length > 512:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 512 instead. You can change that default value by passing --max_seq_length xxx."
                )
                self.max_seq_length = 512
        else:
            if self.args.max_seq_length > self.tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({self.args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length) 
        self.config.vocab_size=len(self.tokenizer)
        print('......................   tokenizer..............................................')
        print(self.tokenizer)
        print("................................................................................")
 

    def process_datasets(self, save_path):
        raw_datasets = None
        if self.args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.args.dataset_name, self.args.dataset_config_name, cache_dir=self.args.cache_dir
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    cache_dir=self.args.cache_dir,
                )
                raw_datasets["train"] = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    cache_dir=self.args.cache_dir,
                )
        else:
            data_files = {}
            if self.args.train_file is not None:
                data_files["train"] = self.args.train_file
            if self.args.validation_file is not None:
                data_files["validation"] = self.args.validation_file
            extension = self.args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.args.cache_dir)

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    cache_dir=self.args.cache_dir,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    cache_dir=self.args.cache_dir,
                )


        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names

        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenizer = self.tokenizer
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_attention_mask=False)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.args.processing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # expanded_inputs_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= self.expanded_inputs_length:
                total_length = (total_length // self.expanded_inputs_length) * self.expanded_inputs_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + self.expanded_inputs_length] for i in range(0, total_length, self.expanded_inputs_length)]
                for k, t in concatenated_examples.items()
            }
            return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.args.processing_num_workers,
            load_from_cache_file=not self.args.overwrite_cache,
            desc=f"Grouping texts in chunks of {self.max_seq_length}",
        )
        
        result_ds = ()
        if "train" in tokenized_datasets:
            train_dataset = tokenized_datasets["train"]
            result_ds += (train_dataset,)
            torch.save(train_dataset, save_path+str(self.max_seq_length)+'train_data.pt') 

        if "validation" in tokenized_datasets:
            eval_dataset = tokenized_datasets["validation"]
            result_ds += (eval_dataset,)

            torch.save(eval_dataset, save_path+str(self.max_seq_length)+'valid_data.pt')

        if "test" in tokenized_datasets:
            test_dataset = tokenized_datasets["test"]
            result_ds += (test_dataset,)

            torch.save(eval_dataset, save_path+str(self.max_seq_length)+'test_data.pt')


        # cach the dataset, so we can load it directly for training

        
        return train_dataset, eval_dataset, test_dataset



    def _create_dataloders(self) -> tuple:

        """
        - a train dataloader that generates batches of src and tgt data
        - a dev dataloader that generates batches of src and tgt data
        """
        if hvd.local_rank() == 0:
            self.logger.info(f"Build train and dev dataloaders ...")
        per_worker_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
##############################################################################################
        if self.args.train_file_path != None and self.args.valid_file_path != None:
            print('loading data...')
            train_dataset  = torch.load(self.args.train_file_path)
            dev_dataset = torch.load(self.args.valid_file_path)
            print('loading done...')    
        elif self.args.dataset_name is not None or self.args.train_file is not None: 
            datasets = self.process_datasets("data/")  
            train_dataset, dev_dataset, test_dataset =   datasets
        else:
            raise ValueError(
                "No available datasets. You need to load a cashed dataset or process a dataset."
            )  
###############################################################################################
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=per_worker_batch_size, #elf.args.batch_size,
            num_workers=0,  # num
            collate_fn=self.collate,
            worker_init_fn=np.random.seed(42),
            pin_memory=True,
        )
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_dataset, num_replicas=hvd.size(), rank=hvd.rank())

        dev_dataloader = DataLoader(
            dev_dataset,
            sampler=dev_sampler,
            batch_size=per_worker_batch_size,
            num_workers=0,
            collate_fn=self.collate,
            worker_init_fn=np.random.seed(42),
            pin_memory=True,
        )

        all_data = (train_dataloader, dev_dataloader)

        return all_data

    def _build_model(self) -> None:
        """
        Build model and update its configuration.
        """
        model_class = get_cls_by_name(self.args.model_name)
        self.model = model_class(self.config) 
        self.model.config = self.config
        self.model = self.model.cuda()




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

    
    
    def _evaluate_dev_loss_ppl(
        self, dev_dataloader: DataLoader,
            epoch: int,
            counter: int,
                ) -> tuple:
        """
        Evaluate parallel dev dataset, calculate evaluation loss, write hypothesis to file, and displays bleu score.
        """
        eval_loss = []
        self.model.eval()
#         stat_selector = torch.zeros(self.config.num_input_sent)

        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                batch_loss = 0
                source = batch['input_ids'].cuda()
                target_ids = batch["labels"].cuda()

                for i in range(0, source.shape[0], self.args.batch_size):
                    outputs = self.model(input_ids=source[i: i + self.args.batch_size],
                                         labels=target_ids[i: i + self.args.batch_size],
                                         output_hidden_states=True,
                                         output_attentions=True,
                                         return_dict=True, )
                    if self.args.fp16 and self.args.apex_opt_lvl == 'O2':
                        loss = outputs['loss']
                    else:
                        loss = outputs.loss
                    loss = loss / self.args.gradient_accumulation_steps
                    batch_loss += loss.detach().item()
                eval_loss += [batch_loss] 
            print("1: eval_loss.......................................", len(eval_loss))
            eval_loss = list(itertools.chain.from_iterable(hvd.allgather_object(eval_loss)))
            print("2: eval_loss.......................................", len(eval_loss))

            eval_loss = np.mean(eval_loss)
            perplexity = math.exp(eval_loss)

          
            return  eval_loss, perplexity


    def _save_model(
        self, optimizer: torch.optim.Optimizer, epoch: int, counter:int, loss: float, scheduler) -> None:
        """
        Args:
            optimizer: optimizer to be saved
            epoch:     epoch at which the model will be saved
            loss:      saved for later comparisons when load and train

        Returns: None

        """
        state = {
            "epoch": epoch,
            "loss": loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler":scheduler.state_dict()

          }
        if self.args.fp16:
            state['amp'] = self.amp.state_dict()
        model_path = os.path.join(self.experiment_path, str(epoch)+"_model.pt")
        torch.save(state, model_path)

    def _use_apex(self, optimizer):
        self.amp = None
        if self.args.fp16:
            try:
                import pathlib
                self.amp = importlib.import_module('apex.apex.amp')
            except ImportError:
                raise ImportError('Install NVIDIA APEX to use fp16 training! Check README.md for instructions.')
        if self.args.fp16:
            self.model, optimizer = self.amp.initialize(self.model, optimizer, enabled=self.args.fp16, opt_level=self.args.apex_opt_lvl)

        return optimizer

    def _load_checkpoint(self, optimizer, scheduler):
        # todo: use iteration number to restore position in dataset?
        # todo: if there is checkpoint in model_path load model from the latest checkpoint (init_checkpoint is None)
        checkpoint = torch.load(self.args.init_checkpoint)
        self.model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if 'amp' in checkpoint and self.args.fp16:
            self.amp.load_state_dict(checkpoint['amp'])
        init_iteration = checkpoint.get('epoch', 0)
        if hvd.local_rank() == 0:
            self.logger.info(f'Model was loaded from: {self.args.init_checkpoint}')
        return init_iteration, optimizer, scheduler


    def train(self) -> None:
        """
        Train model on parallel dataset, evaluate dev data and save best model according to ppl if
        specified.
        """
        start = time.time()
        self.args.working_dir = str(Path(self.args.working_dir).expanduser().absolute())
        os.chdir(self.args.working_dir)
        shutil.copyfile(os.path.join(self.args.working_dir,os.path.basename(__file__)), os.path.join(self.experiment_path, os.path.basename(__file__)))
        self.tb_writer = None
        

        # 1. if CUDA_VISIBLE_DEVICES is not set make all gpus visible
        if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
            print("if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None")
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])
        # 2. init horovod
        hvd.init()

        # 3. prepare config file of the experiement and initialize the summary writer
        if hvd.local_rank() == 0:
            self.logger.info(f'hvd size: {hvd.size()}')
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
            args_dict['HVD_SIZE'] = hvd.size()
            file_path = 'args_config_'+str(self.args.batch_size)+"_"+str(self.args.lr)+'.json'
            json.dump(args_dict, open(self.experiment_path / file_path, 'w'), indent=4)
            log_dir = self.args.experiment_path+'runs_'+batch_string+"_"+str(self.args.lr)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        

        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[hvd.local_rank()]       
        if hvd.local_rank() == 0:
            self.logger.info("Building tokenizer")
        self._create_tokenizer()
        if hvd.local_rank()==0:
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
        self.expanded_inputs_length, self.targets_length = compute_input_and_target_lengths(
            inputs_length=self.max_seq_length,
            noise_density=self.args.mlm_probability,
            mean_noise_span_length=self.args.mean_noise_span_length,
        )

        # 5. Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none
        if hvd.local_rank() == 0:
            self.logger.info(f"Experiment Output Path: \n {self.experiment_path}")
            self.logger.info("Building model...")
        # 6. building the model
        self._build_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        file_path = 'config_model.json'
        self.model.config.to_json_file(self.experiment_path / file_path)
        self.collate = DataCollatorForT5MLM(
        tokenizer=self.tokenizer,
        noise_density=self.args.mlm_probability,
        mean_noise_span_length=self.args.mean_noise_span_length,
        input_length=self.max_seq_length,
        target_length=self.targets_length,
        pad_token_id=self.config.pad_token_id,
        decoder_start_token_id = self.model.config.decoder_start_token_id
    )

        # 7. Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        if hvd.local_rank() == 0:
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

        # init_iteration = 0
        # 13. Create data loaders
        train_dataloader, dev_dataloader = self._create_dataloders()
        
        # 8. build optimizer
        optimizer = self._build_optimizer()
        # 9. Horovod: wrap optimizer with DistributedOptimizer.
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=self.model.named_parameters(),
                                             compression=compression,
                                             op=hvd.Average,
                                             gradient_predivide_factor=1.0,
                                             backward_passes_per_step=self.args.gradient_accumulation_steps,
                                             )

        # 10. scheduler
        # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW.eps
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py        
        
        num_update_steps_per_epoch =len(train_dataloader)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epochs * num_update_steps_per_epoch
        else:
            self.args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if hvd.local_rank() == 0:
            self.logger.info(f'num_update_steps_per_epoch: {num_update_steps_per_epoch}')
            self.logger.info(f'self.args.max_train_steps: {self.args.max_train_steps}')
            self.logger.info(f'self.args.epochs: {self.args.epochs}')

        scheduler = get_scheduler(
                        self.args.lr_scheduler_type,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.warmup_steps,
                        num_training_steps=self.args.max_train_steps,
                    )
        # 11. use Apex
        optimizer = self._use_apex(optimizer)
        # 12. Load checkpoint if there is
        epoch = 0
        init_point = 0


        if self.args.init_checkpoint:
            epoch, optimizer, scheduler = self._load_checkpoint(optimizer, scheduler)  #

        epoch = 0
        

        scheduler = get_scheduler(
                        self.args.lr_scheduler_type,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.warmup_steps,
                        num_training_steps=self.args.max_train_steps,
                    )

        #14. Initializing early stopping if it is in config
        if self.config.early_stopping == True:
            early_stopping = EarlyStopping()
        if hvd.local_rank() == 0:
            self.logger.info("Initializing early stopping...")            
       
        pbar = None
        init_point = epoch
        
        if hvd.local_rank() == 0:
            self.logger.info(f"train_dataloader lenght = number of updates per epoch: {len(train_dataloader)}")
            self.logger.info("Start training...\n")
            self.logger.info("-----------------------------------------------------------------------------------------------------------------------------------------------")
            self.logger.info(f'| epoch | batch | learning rate | train loss | eval loss | perplexity  | epoch val loss  | epoch train loss |   batch time   |   epoch time   |')
            self.logger.info("-----------------------------------------------------------------------------------------------------------------------------------------------")
            pbar = tqdm(total=self.args.epochs)
            pbar.update(epoch+1)

        while epoch < self.args.epochs:
            count_update_steps_per_epoch = 0
            epoch_losses = []
            running_loss = []
            counter_val = 0
            total_epoch_loss_val = 0.0
            
            if hvd.local_rank() == 0:
                start_epoch = time.time()
            if epoch > init_point and hvd.local_rank() == 0:
                self._save_model(optimizer, epoch, 0, eval_loss, scheduler)
            start_batch = time.time()
            end_batch = time.time()
#             stat_selector = torch.zeros(self.config.num_input_sent)
            for counter, batch in enumerate(tqdm(train_dataloader)):
            ############################################   Batch  ###############################################
                start_batch = time.time()
                self.model.train()
                self.model = self.model.cuda()
                source = batch['input_ids'].cuda()
                target_ids = batch["labels"].cuda()
                batch_loss = 0
                for j in range(0, source.shape[0], self.args.batch_size):
                    ##################################  Mini Batch  ######################################

                    outputs = self.model(input_ids=source[j: j + self.args.batch_size],
                                                         labels=target_ids[j: j + self.args.batch_size],
                                                         output_hidden_states=True, 
                                                         output_attentions=True,
                                                         return_dict=True, )
                    if self.args.fp16 and self.args.apex_opt_lvl == 'O2':
                        loss = outputs['loss']
                    else:
                        loss = outputs.loss
                    # divide loss on gradient_accumulation_steps to get average loss for sub-batches
                    loss = loss / self.args.gradient_accumulation_steps
                    batch_loss += loss.detach().item()

                    if self.args.fp16:
                        with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                            # last sub-batch, call synchronize within amp.scale_loss scope
                            # mb move to just above with optimizer.skip_synchronize()
                            if j == (len(source) // self.args.batch_size - 1) * self.args.batch_size:
                                optimizer.synchronize()
                    else:
                        loss.backward()
                    ####################################  End Mini Batch I changed the ident and include inside for loop  ########################################


                running_loss += [batch_loss]
                epoch_losses += [batch_loss]
                
                ###  step and zero grade 
                if self.args.fp16:
                    with optimizer.skip_synchronize():
                        optimizer.step()
                else:
                     optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                count_update_steps_per_epoch += 1


                '''
                batch_losses before gathering self.args.batch_size
                batch_losses after gathering self.args.batch_size * number of gpus
                '''

                if counter % self.args.save_interval == 0 and counter > 0:
                    eval_loss, perplexity = self._evaluate_dev_loss_ppl(dev_dataloader,
                                                     epoch + 1,
                                                     len(train_dataloader)*epoch+ counter
                                                     )

                    print("running_loss ......................................................", len(running_loss))
                    train_mean_loss = list(itertools.chain.from_iterable(hvd.allgather_object(running_loss)))  
                    print("running_loss ......................................................", len(train_mean_loss))
                    train_mean_loss = np.mean(train_mean_loss)                    
                    
                    if hvd.local_rank() == 0:    
                        # scheduler.optimizer.param_groups  dict_keys(['params', 'lr', 'eps', 'clip_threshold', 'decay_rate', 'beta1', 'weight_decay', 'scale_parameter', 'relative_step', 'warmup_init', 'initial_lr'])
                        total_epoch_loss_val += eval_loss
                        tqdm.write(
                            "Epoch: %d, Step: %5d, loss: %.3f"
                            % (epoch + 1, counter + 1, train_mean_loss )  # self.args.save_interval)
                        )
                        ##################   tensor board   #################
                        for j, param_group in enumerate(scheduler.optimizer.param_groups): 
                            for p in ['lr', 'scaled_lr']:
                                if p in param_group and param_group[p] is not None:
                                    self.tb_writer.add_scalar(f'{p}/param_group_{j}', param_group[p], len(train_dataloader)*epoch+ counter)                 
                        self.tb_writer.add_scalar('loss/running_train_loss', train_mean_loss ,
                                                   len(train_dataloader) * epoch + counter)
                        self.tb_writer.add_scalar('loss/valid', eval_loss, len(train_dataloader) * epoch + counter)
                        self.tb_writer.add_scalar('perplexity',  perplexity, len(train_dataloader) * epoch + counter)
                        self.tb_writer.flush()

                        #####################                         logging              #################### 
                        end_batch = time.time()
                        #self.logger.info(f'| epoch | batch | learning rate | train loss | eval loss | perplexity  | epoch val loss  | epoch train loss | batch time | epoch time |')
                        self.logger.info(f'|{epoch:7.0f}|{counter:7.0f}|{scheduler.optimizer.param_groups[0]["lr"]:15.4f}|{train_mean_loss:12.4f}|{eval_loss:11.4f}|{perplexity:13.4f}|                 |                  |{str(datetime.timedelta(seconds=end_batch - start_batch)):16s}|                |')

                        
                    print("total_epoch_loss_val: ", total_epoch_loss_val)
                    running_loss = []
            ####################################################   End Batch  ###############################################################

            '''
            epoch_losses before gathering self.args.batch_size
            epoch_losses after gathering  = epoch_losses before gathering * number of gpus
            '''
            
            eval_loss, perplexity = self._evaluate_dev_loss_ppl(dev_dataloader,
                                 epoch + 1,
                                 len(train_dataloader)*epoch+ len(train_dataloader)
                                 )
            print("running_loss ......................................................", len(running_loss))
            train_mean_loss = list(itertools.chain.from_iterable(hvd.allgather_object(running_loss)))  
            print("running_loss ......................................................", len(train_mean_loss))
            train_mean_loss = np.mean(train_mean_loss)
            
            epoch_losses = list(itertools.chain.from_iterable(hvd.allgather_object(epoch_losses)))

            if len(train_dataloader) % self.args.save_interval == 0:
                counter_val = len(train_dataloader) / self.args.save_interval
            else:
                counter_val = (len(train_dataloader) // self.args.save_interval) + 1
            


            if hvd.local_rank() == 0:
                total_epoch_loss_val += eval_loss
                for j, param_group in enumerate(scheduler.optimizer.param_groups): 
                    for p in ['lr', 'scaled_lr']:
                        if p in param_group and param_group[p] is not None:
                            self.tb_writer.add_scalar(f'{p}/param_group_{j}', param_group[p], len(train_dataloader) * epoch + len(train_dataloader)) 

                self.tb_writer.add_scalar('loss/running_train_loss', train_mean_loss ,
                                           len(train_dataloader) * epoch + len(train_dataloader))
                self.tb_writer.add_scalar('perplexity', perplexity, len(train_dataloader) * epoch + len(train_dataloader))
                self.tb_writer.add_scalar('loss/valid', eval_loss, len(train_dataloader) * epoch + len(train_dataloader))
                self.tb_writer.add_scalar(f'Epoch/average training Loss ', np.mean(epoch_losses), epoch + 1)
                self.tb_writer.add_scalar(f'Epoch/average val loss ', total_epoch_loss_val/counter_val, epoch + 1)  
                self.tb_writer.add_scalar(f'Epoch/val loss after each epoch ', eval_loss, epoch + 1)  

                print("total_epoch_loss_val/counter_val: ", total_epoch_loss_val, counter_val)
                print("len(train_dataloader) , self.args.save_interval: ", len(train_dataloader) , self.args.save_interval)
                self.tb_writer.flush()
                end_batch = time.time()
                end_epoch = time.time()
                #self.logger.info(f'| epoch | batch | train loss | eval loss | perplexity  | epoch val loss  | epoch train loss | batch time | epoch time |')                
                self.logger.info(f'|{epoch:7.0f}|{len(train_dataloader):7.0f}|{scheduler.optimizer.param_groups[0]["lr"]:15.4f}|{train_mean_loss:12.4f}|{eval_loss:11.4f}|{perplexity:13.4f}|{total_epoch_loss_val/counter_val:17.4f}|{np.mean(epoch_losses):18.4f}|{str(datetime.timedelta(seconds=end_batch - start_batch)):16s}|{str(datetime.timedelta(seconds=end_epoch - start_epoch)):16s}|')
                self.logger.info('===============================================================================================================================================')
                pbar.update(1)
                start_epoch = time.time()
            epoch += 1
            print("count_update_steps_per_epoch; ", count_update_steps_per_epoch)
            #################################################     End Epoch  #################################################################



        if hvd.local_rank() == 0:
            pbar.close()
            end = time.time()
            self.logger.info(f"Training done at epoch{epoch}! All outputs saved in {self.args.experiment_path}."
                             f"\nExperiement time was {datetime.timedelta(seconds=end-start)} minutes." )
            
            



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_path', type=str, default='hvd/1_chunk/',
                        help='path where to save model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path with the sharded data in text format')
    parser.add_argument('--train_file_path', type=str, default=None,
                        help='path of processed train data')
    parser.add_argument('--valid_file_path', type=str, default=None,
                        help='path of processed test data')
    
    parser.add_argument('--train_file', type=str, default=None,
                        help='The input training data file (a text file) to be processed.')
    parser.add_argument('--valid_file', type=str, default=None,
                        help='input evaluation data file to evaluate the perplexity on (a text file).')
    ##################################################
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                        help='The name of the dataset to use (via the datasets library).')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1',
                        help='The configuration name of the dataset to use (via the datasets library).')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model.')
    parser.add_argument('--processing_num_workers', type=int, default=1,
                        help='The number of processes to use for the preprocessing.')
    parser.add_argument('--overwrite_cache', type=bool, default=None,
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--mean_noise_span_length', type=float, default = 3.0, help='Mean span length of masked tokens.')
    parser.add_argument('--cache_dir', type=str, default = None, help='Where do you want to store the pretrained models downloaded from s3.')

    ##################################################

    parser.add_argument('--save_interval', type=int, default=10, help='save model every steps')
    parser.add_argument('--working_dir', type=str, default='.',
                        help='working dir, should be a dir with t5-experiments repo (default: .)')
        

    # model args
    parser.add_argument('--model_size', type=str, default='t5-small',
                        help='model_size specifies the base model name (from huggingface) (default: t5-small)')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer_path specifies the base tokenizer name (from huggingface) (default: t5-small)')

    parser.add_argument('--config_dir', type=str,default=None,
                        help='path to model configuration file (default: None)')  #  T5Mem_config.json,  t5-small
    parser.add_argument('--model_name', type=str, default=None,  # Models.T5MemModel:T5MemForConditionalGeneration  or transformers:T5ForConditionalGeneration
                        help='model class name to use (default: transformers:T5ForConditionalGeneration)')
############################ to finetune here put the checkpoint
    parser.add_argument('--init_checkpoint', type=str, 
                        help='path to init checkpoint to load a model from (default: None).')


    # training args
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: None)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 20)')
    parser.add_argument('--max_train_steps', type=int, default=None, help='max_train_steps')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training steps (i.e., gradient updates) (default: 100).')
    parser.add_argument('--patience', type=int, default=10,
                        help='number of epochs with no improvement after which learning rate will be reduced (default: 3).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of batches to accumulate gradients for each worker; it multiplies total batch size.')
    parser.add_argument('--reduction_factor', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=0.00000001, help='learning rate (default: 0.1)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--fp16', action='store_true', default=False, help='use torch.amp for fp16 training')
    parser.add_argument('--apex_opt_lvl', type=str, default='O1', help='apex opt level, O1, O2. (default: O1)')
    parser.add_argument('--optimizer', type=str, default='Adafactor',
                        help='optimizer name: AdamW, Adafactor. (default: AdamW)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='optimizer weight decay (default: 0.0)')
    parser.add_argument('--scale_parameter', action='store_true', default=False,
                        help='Adafactor scale_parameter (default: False)')
    parser.add_argument('--relative_step', action='store_true', default=False,
                        help='Adafactor relative_step (default: False)')
    parser.add_argument('--warmup_init', action='store_true', default=True,
                        help='Adafactor warmup_init (default: False)')
    parser.add_argument('--warmup_steps', action='store_true', default=2000,
                        help='number of warm up steps')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help= "Ratio of tokens to mask for span masked language modeling loss")
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument('--evaluate_dev', action='store_true', default=True,
                        help='Evaluating during training process(default: True)')
#     parser.add_argument('--max_seq_length', type=int, default = 512,
#                         help='The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model.')
    
    
    

    
    args = parser.parse_args()
    return args

    
def main():
    import logging

    # Create a logging instance
    logger = logging.getLogger('my_application')
    logger.setLevel(logging.INFO) # you can set this to be DEBUG, INFO, ERROR

    # Assign a file-handler to that instance
    fh = logging.FileHandler("file_dir.txt")
    fh.setLevel(logging.INFO) # again, you can set this differently
    # Add the handler to your logging instance
    logger.addHandler(fh)
    # load args
    args = parse_args()
    # load config
    print("********************************************************************")
    if args.config_dir:
        args.config_path = args.config_dir + '/t5-small.json'
    else:
        args.config_path = None

    
    print(args)
    
    print("********************************************************************")
    config = load_config(args)
    print("******************************* config **********************************")
    print(config)
    trainer = Trainer(config, args)
    try:
        trainer.train()
    except:
        with open(args.experiment_path+"exceptions.log", "a") as logfile:
            traceback.print_exc(file=logfile)
        raise
if __name__ == "__main__":
    main()