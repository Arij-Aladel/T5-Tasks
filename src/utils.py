import numpy as np
from typing import Dict, List
import importlib
import logging
from logging import Logger
from transformers import T5Config
from transformers import (T5Tokenizer, BatchEncoding)
import sacrebleu
from sacrebleu import raw_corpus_bleu
from sacrebleu.metrics import BLEU, CHRF, TER

def create_logger(log_file: str) -> Logger:
    """
    Create logger for logging the experiment process.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)

    return logger


def load_config(args):
    """
    loads json configuration file.
    """
    # if config_path then model should be given too
    if not args.config_path:
        print("config from pretrained........................................")
        conf_file = T5Config.from_pretrained(args.model_size)
    else:
        conf_file = T5Config.from_json_file(json_file=args.config_path)
    return conf_file 

def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.
    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class
    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
#     print("module_name, cls_name", module_name, cls_name)
    return getattr(importlib.import_module(module_name), cls_name)


def calculate_bleu_scores(hyps: list, refs: list) -> float:
    """
    calculates bleu score.
    """
    assert len(refs) == len(hyps), "no of hypothesis and references sentences must be same length"
    raw_bleu = raw_corpus_bleu(hyps, [refs]).score
    corpus_bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    # t5_bl = t5_bleu(hyps, refs)['bleu']
    # bleu = BLEU()
    chrf = CHRF()
    ter = TER()
    # bleu_score = bleu.corpus_score(hyps, [refs]).score
    chrf_score = chrf.corpus_score(hyps, [refs]).score
    ter_score = ter.corpus_score(hyps, [refs]).score
    # return raw_bleu, corpus_bleu, bleu_score, chrf_score,ter_score#, t5_bl
    return raw_bleu, corpus_bleu, chrf_score,ter_score#, t5_bl

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.00000001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        print(" From early stopping val_loss: ", val_loss, "self.best_loss:", self.best_loss)
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
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
    def __init__(self, tokenizer: T5Tokenizer,
                noise_density: float,
                mean_noise_span_length: float,
                input_length: int,
                target_length: int,
                pad_token_id: int):
        
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length= target_length
        self.pad_token_id= pad_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.target_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

#         # to check that tokens are correctly proprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
#         batch["decoder_input_ids"] = shift_tokens_right(
#             batch["labels"], self.pad_token_id, self.decoder_start_token_id
#         )

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
        sentinel_ids = np.where(sentinel_ids != 0, (sentinel_ids + self.tokenizer.vocab_size - 1), 0)
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
    
def my_collate(batch):
    temp_batch={}
    temp_batch['source'] = [b['source'] for b in batch]
    temp_batch['source_ids'] = torch.stack([b['source_ids'] for b in batch])
    temp_batch['source_mask'] = torch.stack([b['source_mask'] for b in batch])
    temp_batch['target'] = [b['target'] for b in batch]
    temp_batch['target_ids'] = torch.stack([b['target_ids'] for b in batch])
    temp_batch['target_mask'] = torch.stack([b['target_mask'] for b in batch])
    temp_batch['source_sents'] = [b['source_sents'] for b in batch] 
    temp_batch['target_sents'] = [b['target_sents'] for b in batch]
    temp_batch['q_id'] = [b['id'] for b in batch]
    temp_batch['facts'] = [b['facts'] for b in batch]

    return temp_batch