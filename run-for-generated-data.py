import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("modelname", default=None, type=str)
parser.add_argument("testlist", default=None, type=str)

args = parser.parse_args()

modelname = args.modelname
testlist = args.testlist

print("RUNNING EXPERIEMNTS FOR {}".format( modelname))


print('Getting top k predictions and probabilities from the {} for GENERATED DATASET {}'.format(modelname, testlist))

os.system('python get_model_reponses.py \
--inputdir processed_datasets/generated_data/ \
--modelname {0}  \
--outputdir outputs/generated_data/{0} \
--testlist {1}'.format(modelname, testlist)
)

if testlist == 'negsimp':
    dataset = '--negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv'
elif testlist == 'role':
    dataset = '--role_stim datasets/ROLE-88/ROLE-88.tsv'
elif testlist == 'negnat':
    dataset = '--negnat_stim datasets/NEG-88/NEG-88-NAT.tsv'

print('Getting prediction accuracy for GENERATED DATASET {}'.format(testlist))
os.system('python prediction-accuracy-generated-data.py {} {}'.format(modelname, testlist))