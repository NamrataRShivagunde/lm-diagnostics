import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("modelname", default=None, type=str)
args = parser.parse_args()

modelname = args.modelname

if not os.path.exists('processed_datasets/'):
    print("Processing the datasets")
    os.system('python proc_datasets.py \
    --outputdir processed_datasets/ \
    --role_stim datasets/ROLE-88/ROLE-88.tsv \
    --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
    --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
    --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv \
    --add_mask_tok')
else:
    print("Data is already processed")

print('Getting top k predictions and probabilities from the {}'.format(modelname))
os.system('python get_model_reponses.py \
--inputdir processed_datasets/ \
--modelname {0}  \
--outputdir outputs/{0}'.format(modelname))

print('Getting prediction accuracy')
os.system('python prediction_accuracy_tests.py \
  --preddir outputs/{0}/ \
  --resultsdir result-acc-sentivity/{0} \
  --models {0} \
  --k_values 5 \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv'.format(modelname))

print('Getting sensitivity results')
os.system('python sensitivity_tests.py \
   --probdir outputs/{0}/ \
   --resultsdir result-acc-sentivity/{0}/sensitivity \
   --models {0} \
   --role_stim datasets/ROLE-88/ROLE-88.tsv \
   --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
   --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
   --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv'.format(modelname))