# Life after BERT: What do Other Muppets Understand about Language? 

This repository is extending the experiment from BERT to other models as mentioned in our paper 'Life after BERT: What do Other Muppets Understand about Language?' on Psycholinguistic Data from Ettinger (2020).

The codebase can run following experiments:
- Evaluate individual model on CPRAG-102, ROLE-88, NEGSIMP-136 datasets
- Generate dataset for NEG-SIMP 
- Evaluate any of these models on Generated NEGSIMP and Generated ROLE datasets.


# Installation

       python -m pip install -r requirements.txt

# Run

## 1. To evaluate a model on datasets from Ettinger (2020)

The codebase supports
- BERT - `bert-base-uncased` , `bert-large-uncased`
- RoBERTa -  `roberta-base` , `roberta-large`
- DistilBERT - `distilbert-base-uncased`
- AlBERT - `albert-base-v1`, `albert-large-v1`, `albert-xl-v1` ,`albert-xxl-v1`
          `albert-base-v2`, `albert-large-v2`, `albert-xl-v2` ,`albert-xxl-v2`
- GPT2 - `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`


      python run.py <modelname>

for example

      python run.py bert-base-uncased

- The model predictions are located at `outputs/<modelname>/`
- The prediction accuracy and sensitivity results are located at `result-acc-sentivity/<modelname>/`
  

## 2. To evaluate all models 
      python run-all-models.py

This will execute the evaluation for all mentioned models and there is no need to specify individual models

## 3. To generate NEG-SIMP dataset

NEGSIMP-136 dataset is extended using```neg-simp-categories.csv``` which contains 56 categories and their subcategories from original paper Battig(1969). 

      python generate-data.py --dataset 'negsimp'

This will use the ```neg-simp-categories.csv``` aqnd generate neg-simp dataset. We call thi sdataset NEG-SIMP-GENERATED

To run one model for generated dataset 

      python run-for-generated-data.py <modelname> 'negsimp'

for example

      python run-for-generated-data.py bert-base-uncased 'negsimp'

to run all models for the generated dataset

      python run-all-models.py \
      --generated-data True \
      --testlist 'negsimp'

 
For point 1, one model is run for all there datasets. But eveluating each model involves each steps, its optional to run these commands. T

## 4. Run individual steps
### Step 4.1 : Process datasets to produce inputs for LM

The raw datasets are in `datasets` folder. proc_datasets.py will take these datasets and convert into XXX-contextlist and XXX-targetlist files. For example - Role-88 dataset will be processed and saved as `role-contextlist` and `role-targetlist`  and is saved in the `output` folder.

Basic usage:
```
python proc_datasets.py \
  --outputdir <location for output files> \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv \
  --add_mask_tok

for example
python proc_datasets.py \
  --outputdir processed_datasets/ \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv \
  --add_mask_tok

```

After running this script, there will be eight files created in the `processed_datasets` folder-

- For ROLE88 - `role-contextlist` , `role-targetlist`, 

- For NEG-88 -`negsimp-contextlist`, `negsimp-contextlist`,  `negnat-contextlist`, `negnat-contextlist`, 

- For CPRAG-34 - `cprag-contextlist`, `cprag-contextlist`, 


### Step 4.2: Get LM predictions/probabilities

Once the dataset is processed and stored in `processed_datasets`, it will be fed into different models to get the predictions and probabilities.

```
python get_model_responses.py  --inputdir <location of folder with processed datasets> \
--modelname <modelname> \
--outputdir <location of folder for output>
```
 for example
 ```
 python get_model_reponses.py --inputdir processed_datasets/ \
 --modelname bert-base-uncased  \
 --outputdir outputs/bert-base
 ```

After executing this, there will be eight files at the location `outputs/<modelname>/`, for example  `outputs/bert-base/`. Out of eight files four are for model's top k predictions and four files are for token probability.   

At this location `outputs/bert-base-uncased/` , the output prediction files will be (for example for bert-base-uncased model)

        modelpreds-cprag-bert-base-uncased  
        modelpreds-negnat-bert-base-uncased 
        modelpreds-negsimp-bert-base-uncased 
        modelpreds-role-bert-base-uncased 

The output probabilities will be stored in 

        modeltgtprobs-cprag-bert-base-uncased
        modeltgtprobs-negnat-bert-base-uncased
        modeltgtprobs-negsimp-bert-base-uncased
        modeltgtprobs-role-bert-base-uncased

 

### Step 4.3: Run accuracy and sensitivity tests for each datasets

To get prediction accuracy for one model on all datasets

    python prediction_accuracy_tests.py \
    --preddir <location of top k prediction and  probabilities output folder> \
    --resultsdir <location of the result folder> \
    --models <modelname> \
    --k_values <k value> \
    --role_stim datasets/ROLE-88/ROLE-88.tsv \
    --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
    --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
    --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv

for example

      python prediction_accuracy_tests.py \
      --preddir outputs/bert-base-uncased/ \
      --resultsdir result-acc-sentivity/bert-base-uncased \
      --models bert-base-uncased \
      --k_values 5 \
      --role_stim datasets/ROLE-88/ROLE-88.tsv \
      --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
      --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
      --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv

This will store prediction accuracy for the specified model at the location `result-acc-sentivity/<modelname>/`


  For sensitivity

   ```
  python sensitivity_tests.py \
    --probdir <location of top k prediction and  probabilities output folder> \
    --resultsdir <location of the result folder> \
    --models <modelname> \
    --role_stim datasets/ROLE-88/ROLE-88.tsv \
    --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
    --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
    --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv
  ``` 
for example
 ```
  python sensitivity_tests.py \
    --probdir outputs/bert-base/ \
    --resultsdir result-acc-sentivity/bert-base/sensitivity \
    --models bert-base-uncased \
    --role_stim datasets/ROLE-88/ROLE-88.tsv \
    --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
    --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
    --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv
  ``` 

This will store sensitivity for the specified model at the location `result-acc-sentivity/sensitivity/<modelname>/`


  # References
Ettinger, A. (2020). What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models. Transactions of the Association for Computational Linguistics, 8, 34-48.

Battig, William F. and William Edward Montague. “Category norms of verbal items in 56 categories A replication and extension of the Connecticut category norms.” Journal of Experimental Psychology 80 (1969): 1-46.