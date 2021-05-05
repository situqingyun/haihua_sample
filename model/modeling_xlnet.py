from transformers.models.xlnet.modeling_xlnet import *
from model.modeling_bert import *
from model.DUMA_util import DUMA

# DUMA
# class DUMA(nn.Module):
#     def __init__(self, config):
#         super(DUMA, self).__init__()
#         beert_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=4)
#         self.attention = BertSelfAttention(beert_config)
#         self.pooler = MeanPooler(beert_config)
#         self.outputlayer = BertSelfOutput(beert_config)
#
#     def forward(self, sequence_output, doc_len, ques_len, option_len, attention_mask=None):
#         doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
#             sequence_output, doc_len, ques_len, option_len)
#         doc_encoder = self.attention(doc_seq_output, encoder_hidden_states=ques_option_seq_output, attention_mask=attention_mask)
#         ques_option_encoder = self.attention(ques_option_seq_output, encoder_hidden_states=doc_seq_output, attention_mask=attention_mask)
#         # fuse: summarize
#         # output = doc_encoder+ques_option_encoder
#         # output = torch.add(doc_encoder, ques_option_encoder)
#
#         doc_pooled_output = self.pooler(doc_encoder[0])
#         ques_option_pooled_output = self.pooler(ques_option_encoder[0])
#         # doc_pooled_output = mean_pooling(doc_encoder, attention_mask)
#         # ques_option_pooled_output = mean_pooling(ques_option_encoder, attention_mask)
#
#         output = self.outputlayer(doc_pooled_output, ques_option_pooled_output)
#
#         # output = self.pooler(output)
#         return output


class XLNetForMultipleChoiceWithDUMA(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, 1)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.duma = DUMA(config)
        # duma = DUMA(config)
        self.dumas = nn.ModuleList([DUMA(config) for _ in range(1)])
        # self.pooler = self.bert.pooler
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        doc_len=None, ques_len=None, option_len=None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # transformer_outputs = self.transformer(flat_input_ids, flat_token_type_ids, flat_attention_mask)

        output = transformer_outputs[0]

        # sequence_output
        sequence_output = output
        for i, duma_module in enumerate(self.dumas):
            sequence_output = duma_module(sequence_output, doc_len, ques_len, option_len, flat_attention_mask)
        output = sequence_output

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
