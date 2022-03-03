import re
import os
import argparse
import random
import copy
import numpy as np
import access_model as tp
import scipy
import scipy.stats
from io import open

def get_model_responses(inputlist,tgtlist,modelname,model,tokenizer,k=5,bert=True):
    if modelname.startswith('gpt') or modelname.startswith('t5'):
        bert = False
    top_preds,top_probs = tp.get_predictions(inputlist,modelname,model,tokenizer,k=k,bert=bert)
    tgt_probs = tp.get_probabilities(inputlist,tgtlist,modelname,model,tokenizer,bert=bert)

    return top_preds,top_probs,tgt_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Get the inputdir for the processed datasets and the huggingface modelname
    parser.add_argument("--inputdir", default='processed_datasets/', type=str)
    parser.add_argument("--modelname",default='bert-base-uncased', type=str)
    parser.add_argument("--outputdir", default=None, type=str)
    parser.add_argument("--testlist", default="cprag,role,negsimp,negnat", type=str)
   
    args = parser.parse_args()

    # testlist = ['cprag','role', 'negsimp','negnat']
    testlist = args.testlist.split(',')

    print('LOADING MODELS')
    model, tokenizer = tp.load_model(args.modelname)

    # For top k predictions
    k = 5

    models = [(args.modelname, model, tokenizer)]

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    for testname in testlist:
        print('testname', testname)
        inputlist = []
        tgtlist = []
        with open(os.path.join(args.inputdir,testname+'-contextlist')) as cont:
            for line in cont: inputlist.append(line.strip())
        with open(os.path.join(args.inputdir,testname+'-targetlist')) as comp:
            for line in comp: tgtlist.append(line.strip())
        for args.modelname,model,tokenizer in models:
            top_preds,top_probs,tgt_probs = get_model_responses(inputlist,tgtlist,args.modelname,model,tokenizer,k=k)
            with open(args.outputdir+'/modelpreds-%s-%s'%(testname,args.modelname),'w') as pred_out:
                for i,preds in enumerate(top_preds):
                    pred_out.write(' '.join(preds))
                    pred_out.write('\n')
            with open(args.outputdir+'/modeltgtprobs-%s-%s'%(testname,args.modelname),'w') as prob_out:
                for i,prob in enumerate(tgt_probs):
                    prob_out.write(str(prob))
                    prob_out.write('\n')
