from transformers import BertPreTrainedModel,BertModel,LxmertPreTrainedModel,LxmertModel
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from losses import *
from Attention import AdaptiveCoFusion
from .utils_ner import valid_sequence_output


class BertSoftmaxNer(BertPreTrainedModel):

    def __init__(self,args,config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.vis2text = nn.Linear(config.visual_dim,config.hidden_size)
        self.loss_type = config.loss_type
        self.mmEncoder = AdaptiveCoFusion(args,config)

    def forward(
        self,
        input_ids=None,
        input_v_features=None,
        valid_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.mmEncoder(sequence_output,input_v_features)

        sequence_output, attention_mask = valid_sequence_output(logits, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfNer(BertPreTrainedModel):

    def __init__(self, args,config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.mmEncoder = AdaptiveCoFusion(args,config)

    def forward(
        self,
        input_ids=None,
        input_v_features=None,
        valid_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decode=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.mmEncoder(sequence_output, input_v_features)

        sequence_output, attention_mask = valid_sequence_output(logits, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if decode:
            tags = self.crf.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs

        return outputs


class LxmertSoftmaxNer(LxmertPreTrainedModel):

    def __init__(self,args,config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.lxmert = LxmertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mmEncoder = AdaptiveCoFusion(args, config)
        self.loss_type = config.loss_type

    def forward(
        self,
        input_ids=None,
        visual_feats=None,
        visual_pos=None,
        valid_mask=None,
        attention_mask=None,
        visual_attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """

        outputs = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.mmEncoder(sequence_output, visual_feats)

        sequence_output, attention_mask = valid_sequence_output(logits, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)



class LxmertCrfNer(LxmertPreTrainedModel):

    def __init__(self,args,config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.lxmert = LxmertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mmEncoder = AdaptiveCoFusion(args, config)
        self.loss_type = config.loss_type

    def forward(
        self,
        input_ids=None,
        visual_feats=None,
        visual_pos=None,
        valid_mask=None,
        attention_mask=None,
        visual_attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decode=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """

        outputs = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.mmEncoder(sequence_output, visual_feats)

        sequence_output, attention_mask = valid_sequence_output(logits, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if decode:
            tags = self.crf.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs

        return outputs