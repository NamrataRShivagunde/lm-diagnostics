import torch
import argparse
import re
import os
import copy
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import transformers

device = "cuda" if torch.cuda.is_available() else "cpu"
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", '']  # nltk

def load_model(modeldir):
    """Loads the tokenizer and model for the given Huggingface model name
    Argument: 
        modeldir (str) : Huggingface model name e.g. 'bert-base-uncased'

    Returns:
        model : Transformer model
        tokenizer : Transformer tokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir)
    if modeldir.startswith('gpt'):
        model = transformers.GPT2LMHeadModel.from_pretrained(modeldir).to(device)
    elif modeldir.startswith('t5'):
        model = transformers.T5ForConditionalGeneration.from_pretrained(modeldir).to(device)
    else: # for bert, roberta, albert
        model = transformers.AutoModelForMaskedLM.from_pretrained(modeldir).to(device)
    model.eval()
    return model,tokenizer


def prep_input(input_sents,modelname,tokenizer,bert=True):
    """ Input sentences are processed according to the model's requirement
    Arguments  
        input_sents (list) : list of input sentences
        modelname (str) : Huggingface model name e.g. 'bert-base-uncased'
        tokenizer : Transformer tokenizer
        bert (bool) : set to False for gpt and t5 models
    Yield 
        masked_index (int) : index of the mask token
        tokenized_text (dict) : output of the tokenizer
    """
    for sent in input_sents:
        masked_index = None 
        if modelname.startswith('roberta'):
            sent = sent.replace('[MASK]','<mask> .') 
            mask_id = tokenizer.convert_tokens_to_ids('<mask>')
        elif modelname.startswith('gpt'):
            mask_id = -1
            sent = sent.replace('[MASK]','')
        elif modelname.startswith('t5'):
            sent = sent.replace('[MASK]', '<extra_id_0>.') # tokenizer adds </s> end of token

        else: # for distillbert, bert, alberta
            mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
            sent = sent + '.'    

        tokenized_text = tokenizer(sent, return_tensors="pt")
    
        # Setting the mask index
        if bert:
            for i,tok in enumerate(tokenized_text['input_ids'].reshape(-1)):
                if tok == mask_id:
                    masked_index = i

        else: # bert is false for gpt, t5
            if modelname.startswith('gpt'): 
                masked_index = -1
            else: # for t5
                masked_index = 1

        yield masked_index, tokenized_text


def get_predictions(input_sents,modelname,model,tokenizer,k=5,bert=True):
    """ Get top k predictions from the model

    Arguments 
        input_sents (list) : list of input sentences
        modelname (str) : Huggingface model name e.g. 'bert-base-uncased'
        tokenizer : Transformer tokenizer
        bert (bool) : set to False for gpt and t5 models

    Returns 
        token_preds (list) : top k predictions
        tok_probs (list) : token probability 
    """
    token_preds = []
    tok_probs = []
    filtered = False

    for mi, tokenized_text in prep_input(input_sents,modelname,tokenizer,bert=bert):
        tokenized_text = tokenized_text.to(device)

        with torch.no_grad():
            if modelname.startswith('t5'):
                decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                predictions = model(input_ids=tokenized_text.input_ids, decoder_input_ids=decoder_ids).logits
            else:
                predictions = model(**tokenized_text).logits
                
        if bert: # for bert, roberta, albert
            softpred = torch.softmax(predictions[0,mi],0)
            top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
            top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
            top_tok_preds = tokenizer.decode(top_inds)
            top_tok_preds = top_tok_preds.split(' ')

        else: # for gpt, t5
            softpred = torch.softmax(predictions[0, mi, :],0)
            top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
            top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
            top_tok_preds = top_inds.strip()
            top_tok_preds.append(tokenizer.decode(top_tok_preds))

            # top_tok_preds = []
            # i = 0
            # while len(top_tok_preds) < k:
            #     if tokenizer.decode(top_inds[i]).strip() not in stop_words:
            #         top_tok_preds.append(tokenizer.decode(top_inds[i]).strip())
            #     i += 1

        top_tok_preds = [re.sub('\<\/w\>','',e) for e in top_tok_preds]

        token_preds.append(top_tok_preds)
        tok_probs.append(top_probs)

    return token_preds,tok_probs

def get_probabilities(input_sents,tgtlist,modelname,model,tokenizer,bert=True):
    """ Get top k predictions from the model

    Arguments 
        input_sents (list) : list of input sentences
        tgtlist (list) : target list
        modelname (str) : Huggingface model name e.g. 'bert-base-uncased'
        tokenizer : Transformer tokenizer
        bert (bool) : set to False for gpt and t5 models

    Returns 
        token_probs (list) : token probabilites
    """
    token_probs = []
    for i,(mi, tokenized_text) in enumerate(prep_input(input_sents,modelname,tokenizer,bert=bert)):
        tokenized_text = tokenized_text.to(device)

        with torch.no_grad():
            if modelname.startswith('t5'):
                decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                predictions = model(input_ids=tokenized_text.input_ids, decoder_input_ids=decoder_ids).logits
            else:
                predictions = model(**tokenized_text).logits
        tgt = tgtlist[i]
        if bert: # for bert, roberta, albert
            softpred = torch.softmax(predictions[0,mi],0)
        else: # for gpt, t5
            softpred = torch.softmax(predictions[0, mi, :],0)

        try:
            tgt_ind = tokenizer.convert_tokens_to_ids([tgt])[0]
        except:
            this_tgt_prob = np.nan
        else:
            this_tgt_prob = softpred[tgt_ind].item()
        token_probs.append(this_tgt_prob)
    return token_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--modeldir", default=None, type=str, required=True)
    args = parser.parse_args()
    get_predictions(args.input_file)
