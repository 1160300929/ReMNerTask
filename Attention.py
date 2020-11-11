import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertPreTrainedModel


import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel,BertModel
from torch.nn import CrossEntropyLoss, MSELoss



class CoAttention(nn.Module):
    def __init__(self, args):
        super(CoAttention, self).__init__()
        self.args = args

        # linear for word-guided visual attention
        self.text_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.img_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(args.hidden_dim * 2, 1)

        # linear for visual-guided textual attention
        self.text_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.img_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, text_features, img_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. Word-guided visual attention ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = text_features.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)
        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.max_seq_length, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = self.text_linear_1(text_features_rep)
        img_features_rep = self.img_linear_1(img_features_rep)
        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([text_features_rep, img_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        visual_att = self.att_linear_1(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_img_features = torch.matmul(visual_att, img_features)  # Vt_hat

        ############### 2. Visual-guided textual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = att_img_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_length, 1)
        text_features_rep = text_features.unsqueeze(1).repeat(1, self.args.max_seq_length, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = self.img_linear_2(img_features_rep)
        text_features_rep = self.text_linear_2(text_features_rep)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([img_features_rep, text_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        textual_att = self.att_linear_2(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_text_features = torch.matmul(textual_att, text_features)  # Ht_hat

        return att_text_features, att_img_features


class GMF(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, att_text_features, att_img_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

        gate_img = self.gate_linear(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img = torch.sigmoid(gate_img)  # [batch_size, max_seq_len, 1]
        gate_img = gate_img.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img, new_img_feat) + torch.mul(1 - gate_img, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features


class FiltrationGate(nn.Module):
    """
    In this part, code is implemented in other way compare to equation on paper.
    So I mixed the method between paper and code (e.g. Add `nn.Linear` after the concatenated matrix)
    """

    def __init__(self, args,label_num):
        super(FiltrationGate, self).__init__()
        self.args = args

        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.multimodal_linear = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

        self.resv_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.output_linear = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

    def forward(self, text_features, multimodal_features):
        """
        :param text_features: Original text feature from BiLSTM [batch_size, max_seq_len, hidden_dim]
        :param multimodal_features: Feature from GMF [batch_size, max_seq_len, hidden_dim]
        :return: output: Will be the input for CRF decoder [batch_size, max_seq_len, hidden_dim]
        """
        # [batch_size, max_seq_len, 2 * hidden_dim]
        concat_feat = torch.cat([self.text_linear(text_features), self.multimodal_linear(multimodal_features)], dim=-1)
        # This part is not written on equation, but if is needed
        filtration_gate = torch.sigmoid(self.gate_linear(concat_feat))  # [batch_size, max_seq_len, 1]
        filtration_gate = filtration_gate.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]

        reserved_multimodal_feat = torch.mul(filtration_gate,
                                             torch.tanh(self.resv_linear(multimodal_features)))  # [batch_size, max_seq_len, hidden_dim]
        output = self.output_linear(torch.cat([text_features, reserved_multimodal_feat], dim=-1))  # [batch_size, max_seq_len, num_tags]

        return output


# the fusion method in
# Adaptive Co-attention Network for Named Entity Recognition in Tweets" 2017 AAAI
class AdaptiveCoFusion(nn.Module):
    def __init__(self,args,config):
        self.args = args
        self.co_attention = CoAttention(args)
        self.gmf = GMF(args)
        self.filtration_gate = FiltrationGate(args,config.num_labels)
    def forward(self, txt_hidden,vis_hidden):
        att_text_features, att_img_features = self.co_attention(txt_hidden, vis_hidden)
        multimodal_features = self.gmf(att_text_features, att_img_features)
        logits = self.filtration_gate(txt_hidden, multimodal_features)
        return logits


class UMTFusion(nn.Module):
    def __init__(self,config):
        pass

