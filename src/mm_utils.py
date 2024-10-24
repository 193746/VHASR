import os
import torch
from transformers import CLIPModel, CLIPProcessor

class Merger():
    def __init__(self,root_path):
        self.preprocessor = CLIPProcessor.from_pretrained(root_path)
        clip_model=CLIPModel.from_pretrained(root_path)
        self.text_model=clip_model.text_model
        self.visual_projection=clip_model.visual_projection
        self.text_projection=clip_model.text_projection
    
    def merge_two_stram_by_m3(self,error_calculator,img_embeds,asr_out,bias_out,gold_out,va_similarity):
        asr_list=error_calculator.convert_to_char_single(asr_out)
        bias_list=error_calculator.convert_to_char_single(bias_out)
        gold_list=error_calculator.convert_to_char_single(gold_out)
        va_similarity=torch.where(va_similarity>=0.8,1,0)

        merge_list=[]
        for j in range(len(asr_list)):
            text=[asr_list[j],bias_list[j]]
            img_embed=img_embeds[j]
            if text[0]==text[1] or va_similarity[j].item()==0: # filter img_wer of low audio-image similarity 
                merge_list.append(text[0])
                continue     
            # cal vision-text similarity
            merge_text=self.cal_similarity(img_embed,text)         
            merge_list.append(merge_text)

        merge_wer=error_calculator.calculate_wer(merge_list,gold_list)
        return merge_wer

    def merge_two_stram_by_m2(self,error_calculator,img_embeds,asr_out,bias_out,gold_out):
        asr_list=error_calculator.convert_to_char_single(asr_out)
        bias_list=error_calculator.convert_to_char_single(bias_out)
        gold_list=error_calculator.convert_to_char_single(gold_out)

        merge_list=[]
        for j in range(len(asr_list)):
            text=[asr_list[j],bias_list[j]]
            img_embed=img_embeds[j]
            if text[0]==text[1]:
                merge_list.append(text[0])
                continue
            merge_text=self.cal_similarity(img_embed,text)         
            merge_list.append(merge_text)

        merge_wer=error_calculator.calculate_wer(merge_list,gold_list)
        return merge_wer

    def cal_similarity(self,img_embed,text):
        text_inputs=self.preprocessor(text=text,return_tensors="pt", padding=True)
        tokens=text_inputs["input_ids"]
        if len(tokens[0])>77 or len(tokens[1])>77:
            return text[0]
        text_embeds=self.text_model(**text_inputs)[0]
        text_embeds=self.text_projection(text_embeds)

        image_embeds=self.visual_projection(img_embed.unsqueeze(dim=0))

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_token = torch.matmul(text_embeds, image_embeds.t()).squeeze(-1)

        best_index=logits_per_token.argmax(dim=0)
        best_tokens=[tokens[best_index[i].item(),i].item() for i in range(len(best_index))]
        merge_text=self.preprocessor.decode(best_tokens[1:-1])
        merge_text=merge_text.replace("<|endoftext|>","")
        merge_text=merge_text.strip()
        return merge_text