import logging
from torchblocks.processor.base import DataProcessor, LOWERCASE_STRS
from torchblocks.processor.utils import InputFeatures
import copy
import json

logger = logging.getLogger(__name__)


class MultipleChoiceProcessor(DataProcessor):
    """
        多选题
    """

    def convert_to_features(self, examples, label_list, max_seq_length):
        '''
        文本分类
        '''
        label_map = {label: i for i, label in enumerate(label_list)} if label_list is not None else {}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            if not isinstance(texts, list):
                raise ValueError(" texts type: expected one of (list,)")

            # encode_dic = tokenizer.encode_plus(
            #     text=para,
            #     text_pair=qa,
            #     max_length=max_length,
            #     padding='max_length',
            #     truncation=True,
            #     pad_to_multiple_of=True,
            #     add_special_tokens=True,
            #     return_attention_mask=True,
            #     return_token_type_ids=True,
            #     return_tensors='pt'
            # )
            # input_ids = encode_dic['input_ids'].tolist()[0]
            # attention_mask = encode_dic['attention_mask'].tolist()[0]
            # token_type_ids = encode_dic['token_type_ids'].tolist()[0]

            # inputs = self.encode(texts, max_seq_length)

            input_ids = list()
            attention_masks = list()
            token_type_ids = list()

            for count, text in enumerate(texts):
                inputs = self.encode(text, max_seq_length)
                input_id = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                token_type_id = inputs['token_type_ids']
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)

            count += 1
            # for count, (input_id, attention_mask, token_type_id) in enumerate(inputs):
            #     input_ids.append(input_id)
            #     attention_masks.append(attention_mask)
            #     token_type_ids.append(token_type_id)

            # processed into four options
            for j in range(4 - count):
                input_id = [0] * max_seq_length
                attention_mask = [0] * max_seq_length
                token_type_id = [0] * max_seq_length

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)

            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_masks
            inputs['token_type_ids'] = token_type_ids

            inputs['guid'] = example.guid
            if example.label_ids is not None:
                label_ids = [0] * len(label_map)  # 多标签分类
                for i, lb in enumerate(example.label_ids):
                    if isinstance(lb, str):
                        label_ids[label_map[lb]] = 1
                    elif isinstance(lb, (float, int)):
                        label_ids[i] = int(lb)
                    else:
                        raise ValueError("multi label type: expected one of (str,float,int)")
                inputs['label_ids'] = label_ids
            if example.label is not None:
                if isinstance(example.label, (float, int)):
                    label = int(example.label)
                elif isinstance(example.label, str):
                    label = label_map[example.label]
                else:
                    raise ValueError("label type: expected one of (str,float,int)")
                inputs['label'] = label
            if ex_index < 5:
                self.print_examples(**inputs)

            features.append(InputFeatures(**inputs))
        return features


class DCMNInputFeatures:
    """
    A single set of features of processor.
    Property names are the same names as the corresponding inputs to a model.
    """

    def __init__(self,
                 input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 label=None,
                 label_ids=None,
                 doc_len=None, ques_len=None, option_len=None,
                 **kwargs):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.label_ids = label_ids
        self.doc_len = doc_len
        self.ques_len = ques_len
        self.option_len = option_len
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DCMNMultipleChoiceProcessor(DataProcessor):

    def get_input_keys(self):
        '''
        inputs输入对应的keys，需要跟模型输入对应
        '''
        keys = ['guid', 'input_ids', 'attention_mask', 'token_type_ids', 'doc_len', 'ques_len', 'option_len']
        if self.encode_mode == 'one':
            return keys + ['labels']
        elif self.encode_mode == 'pair':
            return [f'{LOWERCASE_STRS[i]}_{item}' for i in range(2) for item in keys] + ['labels']
        elif self.encode_mode == 'triple':
            return [f'{LOWERCASE_STRS[i]}_{item}' for i in range(3) for item in keys] + ['labels']

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        pop_label = True
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(1)
            else:
                tokens_b.pop(1)

    def _truncate_seq_pair(self, context_token, ques_token, option_token, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        pop_label = True
        while True:
            total_length = len(context_token) + len(ques_token) + len(option_token)
            mean_length = total_length // 3
            if total_length <= max_length:
                break
            if len(context_token) > mean_length:
                context_token.pop(0)
            elif len(ques_token) > mean_length:
                ques_token.pop(0)
            else:
                option_token.pop(0)
            # if len(tokens_a) > len(tokens_b):
            #     tokens_a.pop(1)
            # else:
            #     tokens_b.pop(1)

    def _truncate_seq_pair_remove_context(self, context_token, ques_token, option_token, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        pop_label = True
        while True:
            total_length = len(context_token) + len(ques_token) + len(option_token)
            # mean_length = total_length // 3
            if total_length <= max_length:
                break
            if len(context_token) > 0:
                context_token.pop(0)
            elif len(ques_token) > 0:
                ques_token.pop(0)
            else:
                option_token.pop(0)
            # if len(tokens_a) > len(tokens_b):
            #     tokens_a.pop(1)
            # else:
            #     tokens_b.pop(1)

    def encode(self, texts, max_seq_length):
        # We create a copy of the context tokens in order to be
        # able to shrink it according to ending_tokens
        context_sentence = texts[2]
        ending = texts[1]
        start_ending = texts[0]
        context_tokens = self.tokenizer.tokenize(context_sentence)  # article
        start_ending_tokens = self.tokenizer.tokenize(start_ending)  # question

        context_tokens_choice = context_tokens[:]  # + start_ending_tokens

        # todo 缺省的选项如何填充
        # 这里使用了tokenizer，tokenize，参考下dissertation
        ending_token = self.tokenizer.tokenize(ending)
        # option_len = len(ending_token)
        # ques_len = len(start_ending_tokens)

        # ending_tokens = start_ending_tokens + ending_token

        # Modifies `context_tokens_choice` and `ending_tokens` in
        # place so that the total length is less than the
        # specified length.  Account for [CLS], [SEP], [SEP] with
        # "- 3"
        # ending_tokens = start_ending_tokens + ending_tokens
        # self._truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

        # self._truncate_seq_pair(context_tokens_choice, start_ending_tokens, ending_token, max_seq_length - 3)
        self._truncate_seq_pair_remove_context(context_tokens_choice, start_ending_tokens, ending_token,
                                               max_seq_length - 3)

        ending_tokens = start_ending_tokens + ending_token  # ending_tokens: question+option    ending_token: option

        doc_len = len(context_tokens_choice)
        # if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
        #     ques_len = len(ending_tokens) - option_len

        option_len = len(ending_token)
        ques_len = len(start_ending_tokens)

        tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
        segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert (doc_len + ques_len + option_len) <= max_seq_length
        if (doc_len + ques_len + option_len) > max_seq_length:
            print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
            assert (doc_len + ques_len + option_len) <= max_seq_length
        # choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))
        return input_ids, input_mask, segment_ids, doc_len, ques_len, option_len

    """
        多选题
    """

    def convert_to_features(self, examples, label_list, max_seq_length):
        '''
        文本分类
        '''
        label_map = {label: i for i, label in enumerate(label_list)} if label_list is not None else {}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            if not isinstance(texts, list):
                raise ValueError(" texts type: expected one of (list,)")

            input_ids = list()
            attention_masks = list()
            token_type_ids = list()
            doc_lens, ques_lens, option_lens = list(), list(), list()

            for count, text in enumerate(texts):
                inputs = self.encode(text, max_seq_length)
                input_id = inputs[0]
                attention_mask = inputs[1]
                token_type_id = inputs[2]
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
                doc_lens.append(inputs[3])
                ques_lens.append(inputs[4])
                option_lens.append(inputs[5])

                if not (inputs[3] >= 0 and inputs[4] >= 0 and inputs[5] >= 0):
                    print('guid:', example.guid)
                    print(inputs[3], inputs[4], inputs[5])

            count += 1

            # processed into four options
            for j in range(4 - count):
                input_id = [0] * max_seq_length
                attention_mask = [0] * max_seq_length
                token_type_id = [0] * max_seq_length

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
                doc_lens.append(0)
                ques_lens.append(0)
                option_lens.append(0)

            inputs = dict()
            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_masks
            inputs['token_type_ids'] = token_type_ids
            inputs['doc_len'] = doc_lens
            inputs['ques_len'] = ques_lens
            inputs['option_len'] = option_lens

            inputs['guid'] = example.guid
            if example.label_ids is not None:
                label_ids = [0] * len(label_map)  # 多标签分类
                for i, lb in enumerate(example.label_ids):
                    if isinstance(lb, str):
                        label_ids[label_map[lb]] = 1
                    elif isinstance(lb, (float, int)):
                        label_ids[i] = int(lb)
                    else:
                        raise ValueError("multi label type: expected one of (str,float,int)")
                inputs['label_ids'] = label_ids
            if example.label is not None:
                if isinstance(example.label, (float, int)):
                    label = int(example.label)
                elif isinstance(example.label, str):
                    label = label_map[example.label]
                else:
                    raise ValueError("label type: expected one of (str,float,int)")
                inputs['label'] = label
            if ex_index < 5:
                self.print_examples(**inputs)

            features.append(DCMNInputFeatures(**inputs))
        return features
