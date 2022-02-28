import torch
import argparse
import re
import os
import copy
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import transformers

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", '']  # nltk

def load_model(modeldir):
    tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir)
    if modeldir.startswith('gpt'):
        model = transformers.GPT2LMHeadModel.from_pretrained(modeldir)
    elif modeldir.startswith('t5'):
        model = transformers.T5ForConditionalGeneration.from_pretrained(modeldir)
    else: # for bert, roberta, albert
        model = transformers.AutoModelForMaskedLM.from_pretrained(modeldir)
    model.eval()
    # model.to('cuda')
    return model,tokenizer

def prep_input(input_sents,modelname,tokenizer,bert=True):
    for sent in input_sents:
        masked_index = None
        # Processing the input sentences according to the model's requirement
        if modelname.startswith('roberta'):
            sent = sent.replace('[MASK]','<mask> .') 
            mask_id = tokenizer.convert_tokens_to_ids('<mask>')
        elif modelname.startswith('gpt'):
            mask_id = -1
            sent = sent.replace('[MASK]','')
        elif modelname.startswith('t5'):
            sent = sent.replace('[MASK]', '<extra_id_0>')
            sent = sent + '</s>'
        else: # for bert, alberta
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

        tokens_tensor = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])
        yield tokens_tensor, masked_index, tokenized_text


def get_predictions(input_sents,modelname,model,tokenizer,k=5,bert=True):
    token_preds = []
    tok_probs = []

    for tokens_tensor, mi, tokenized_text in prep_input(input_sents,modelname,tokenizer,bert=bert):
        tokens_tensor = tokens_tensor#.to('cuda')

        with torch.no_grad():
            if modelname.startswith('t5'):
                decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids#.to(device)
                predictions = model(input_ids=tokenized_text.input_ids, decoder_input_ids=decoder_ids).logits
            else:
                predictions = model(**tokenized_text).logits
                
        if bert: # for bert, roberta, albert
            softpred = torch.softmax(predictions[0,mi],0)
            top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
            top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
            # top_tok_preds = tokenizer.convert_ids_to_tokens(top_inds)

            # if modelname.startswith('roberta') or modelname.startswith('albert'):
            top_tok_preds = tokenizer.decode(top_inds)
            top_tok_preds = top_tok_preds.split(' ')

        else: # for gpt, t5
            softpred = torch.softmax(predictions[0, mi, :],0)
            top_inds = torch.argsort(softpred,descending=True)[:180 + k].cpu().numpy()
            top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
       
            top_tok_preds = []
            i = 0
            while len(top_tok_preds) < k:
                if tokenizer.decode(top_inds[i]).strip() not in stop_words:
                    top_tok_preds.append(tokenizer.decode(top_inds[i]).strip())
                i += 1

        top_tok_preds = [re.sub('\<\/w\>','',e) for e in top_tok_preds]

        token_preds.append(top_tok_preds)
        tok_probs.append(top_probs)

    return token_preds,tok_probs

def get_probabilities(input_sents,tgtlist,modelname,model,tokenizer,bert=True):
    token_probs = []
    for i,(tokens_tensor, mi, tokenized_text) in enumerate(prep_input(input_sents,modelname,tokenizer,bert=bert)):
        tokens_tensor = tokens_tensor#.to('cuda')

        with torch.no_grad():
            if modelname.startswith('t5'):
                print('pass1')
                decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids#.to(device)
                predictions = model(input_ids=tokenized_text.input_ids, decoder_input_ids=decoder_ids).logits
                print('pass2')
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
        print('pass3')
    return token_probs

################################# Roberta #######################################################

def load_model_roberta(modeldir):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir)
    model = transformers.AutoModelWithLMHead.from_pretrained(modeldir)
    model.eval()
    model.to('cuda')
    return model,tokenizer

def prep_input_roberta(input_sents, tokenizer,bert=True):
    for sent in input_sents:
        masked_index = None
        #sent = sent.replace('[MASK]','<mask>') uncomment for roberta
        sent = sent.replace('[MASK]','')
        mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        tokenized_text = tokenizer(sent, return_tensors="pt")
        for i,tok in enumerate(tokenized_text['input_ids'].reshape(-1)):
          if tok == mask_id:
              masked_index = i
        tokens_tensor = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])
        yield tokenized_text, masked_index, tokens_tensor

def get_predictions_roberta(input_sents,model,tokenizer,k=5, bert=False):
    token_preds = []
    tok_probs = []
    for tokenized_text, mi, tokens_tensor in prep_input_roberta(input_sents,tokenizer):
        with torch.no_grad():
          tokenized_text = tokenized_text.to('cuda')
          predictions = model(**tokenized_text)
          predictions = predictions.logits

        predicted_tokens = []
        predicted_token_probs = []

        if bert:
            softpred = torch.softmax(predictions[0,mi],0)
        else:
            # bert is false for gpt2 
            softpred = torch.softmax(predictions[0, -1, :],0) # for gpt2

        top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
        top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]

        top_tok_preds = tokenizer.convert_ids_to_tokens(top_inds)

        if not bert:
            top_tok_preds = [re.sub('\<\/w\>','',e) for e in top_tok_preds]

        token_preds.append(top_tok_preds)
        tok_probs.append(top_probs)

    return token_preds,tok_probs

def get_probabilities_roberta(input_sents,tgtlist,model,tokenizer,bert=False):
    token_probs = []
    for i,(tokens_tensor, mi,_) in enumerate(prep_input_roberta(input_sents,tokenizer,bert=bert)):
        tokens_tensor = tokens_tensor.to('cuda')

        with torch.no_grad():
            predictions = model(**tokens_tensor)
            predictions = predictions.logits
        tgt = tgtlist[i]
        if bert:
            softpred = torch.softmax(predictions[0,mi],0)
        else:
            mi = predictions.shape[1]
            softpred = torch.softmax(predictions[0, mi-1, :],0)
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
    get_predictions_roberta(args.input_file)
