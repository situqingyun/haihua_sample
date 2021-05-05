import torch
from torch.nn import functional as F
from torch import nn

def concat_all_encoders_hidden_states(all_encoder_layers, rnn, linears):
    all_encoder_layers = all_encoder_layers[1:]
    output_list = list()
    for i in range(len(all_encoder_layers)):
        output, (final_hidden_state, final_cell_state) = rnn(all_encoder_layers[i])
        output_list.append(linears[i](output))

    output_tensor = torch.cat(output_list, 2)
    output_tensor = F.softmax(output_tensor, 2)

    all_layers = torch.cat([torch.unsqueeze(i, 2) for i in all_encoder_layers], axis=2)  # 第三维度拼接
    focus = torch.matmul(torch.unsqueeze(output_tensor, axis=2), all_layers)

    sequence_output = torch.squeeze(focus, 2)
    return sequence_output


# import torch
# from torch.nn import functional as F
#
#
# def concat_all_encoders_hidden_states(all_encoder_layers, linears):
#     all_encoder_layers=all_encoder_layers[1:]
#     output_list = list()
#     for i in range(len(all_encoder_layers)):
#         # output, (final_hidden_state, final_cell_state) = rnn(all_encoder_layers[i])
#         output_list.append(linears[i](all_encoder_layers[i]))
#
#     output_tensor = torch.cat(output_list, 2)
#     output_tensor = F.softmax(output_tensor, 2)
#
#     all_layers = torch.cat([torch.unsqueeze(i, 2) for i in all_encoder_layers], axis=2)  # 第三维度拼接
#     focus = torch.matmul(torch.unsqueeze(output_tensor, axis=2), all_layers)
#
#     sequence_output = torch.squeeze(focus, 2)
#     return sequence_output

# def seperate_seq(sequence_output, doc_len, ques_len, option_len):
#     doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     option_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     for i in range(doc_len.size(0)):
#         doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
#         doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
#         ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
#         ques_option_seq_output[i, :ques_len[i] + option_len[i]] = sequence_output[i,
#                                                                   doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[
#                                                                       i] + 1]
#         option_seq_output[i, :option_len[i]] = sequence_output[i,
#                                                doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
#                                                    i] + 2]
#     return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output
#
#
# class MeanPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()
#
#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output
#
#
# # DUMA
# class DUMA(nn.Module):
#     def __init__(self, config):
#         super(DUMA, self).__init__()
#         self.attention = NeZhaSelfAttention(config)
#         self.pooler = MeanPooler(config)
#         self.outputlayer = BertSelfOutput(config)
#
#     def forward(self, sequence_output, doc_len, ques_len, option_len, attention_mask=None):
#         doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
#             sequence_output, doc_len, ques_len, option_len)
#         doc_encoder = self.attention(doc_seq_output, encoder_hidden_states=ques_option_seq_output,
#                                      attention_mask=attention_mask)
#         ques_option_encoder = self.attention(ques_option_seq_output, encoder_hidden_states=doc_seq_output,
#                                              attention_mask=attention_mask)
#         # fuse: summarize
#         # output = doc_encoder+ques_option_encoder
#         # output = torch.add(doc_encoder, ques_option_encoder)
#         doc_pooled_output = self.pooler(doc_encoder[0])
#         ques_option_pooled_output = self.pooler(ques_option_encoder[0])
#
#         output = self.outputlayer(doc_pooled_output, ques_option_pooled_output)
#
#         output = self.pooler(output)
#         # output = mean_pooling(sequence_output, attention_mask)
#         return output
