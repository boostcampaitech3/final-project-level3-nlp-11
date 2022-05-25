

from transformers import (DetrFeatureExtractor, DetrForObjectDetection,
                          VisualBertForQuestionAnswering,VisualBertPreTrainedModel, BertTokenizerFast,VisualBertConfig,
                          VisualBertModel,
                         YolosFeatureExtractor, YolosForObjectDetection, YolosModel,YolosPreTrainedModel,YolosConfig)
from transformers.modeling_outputs import SequenceClassifierOutput

from feature_extract.modeling_frcnn import GeneralizedRCNN
from feature_extract.preprocessing_image import Preprocess
from feature_extract.utils import Config
import torch
import torch.nn as nn




class VQAModel(VisualBertPreTrainedModel):
    def __init__(self,config):
        super(VQAModel,self).__init__(config)
        self.config=config
        self.model = VisualBertModel(config)
        self.hidden_dim = self.model.config.hidden_size

        

        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = self.model.config.num_labels
        self.linear=nn.Linear(2048,self.config.hidden_size)
        
        #num_label 수정필요
        self.cls = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self,input_ids,token_type_ids,attention_mask,image,labels=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        visual_embeds = self.get_visual_embeddings(image=list(image))#.to(device)
        #visual_embeds=self.linear(visual_embeds)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask=torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)
        index_to_gather = attention_mask.sum(1) - 2
        outputs = self.model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids= token_type_ids,
                            visual_embeds = visual_embeds,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids
                            )
                            #labels=labels) 


        sequence_output = outputs[0]
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        )
        pooled_output = torch.gather(sequence_output, 1, index_to_gather)

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        logits=logits.view(-1,self.num_labels)

        #if labels is not None:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )         




    def get_visual_embeddings(self,image):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        images, sizes, scales_yx = self.image_preprocess(image)
        self.frcnn.eval()
        output_dict = self.frcnn(
                images.to(device),
                sizes.to(device),
                scales_yx=scales_yx.to(device),
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
            )
        features = output_dict.get("roi_features")
        return features #(batch,box_num,emb_dim)