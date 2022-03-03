# To run all step 1, 2 and 3 for a model

run 

```python run.py <modelname>```

for example

```python run.py bert-base-uncased```

The script supports
- BERT - `bert-base-uncased` , `bert-large-uncased`
- RoBERTa -  `roberta-base` , `roberta-large`
- DistilBERT - `distilbert-base-uncased`
- AlBERT - `albert-base-v1`, `albert-large-v1`, `albert-xl-v1` ,`albert-xxl-v1`
          `albert-base-v2`, `albert-large-v2`, `albert-xl-v2` ,`albert-xxl-v2`
- GPT2 - `gpt2`, `gpt2-medium`, `gpt22-large`, `gpt2-xl`

# To run all models 
 ``` python run-all-models.py ```


For individual steps follow these steps.

# To run one model for generated datasets

``` python run-for-generated-data.py <modelname> <dataset>```

for example

```python run-for-generated-data.py bert-base-uncased 'negsimp'```

Currently supporting datasets
- 'negsimp'
- 'role'
- 'negnat'

Models are same as above list

to run all models for one generated dataset
 ``` python run-all-models.py --generated-data True --testlist 'negsimp' ```

# To generate dataset

NEG-SIMP is an extension of NEG-SIMP from (ettinger, 2020) using 56 categories and their subcategories from original paper (Battig, 1969). 
To extend the NEG-SIMP dataset run

```python generate-data.py --dataset 'negsimp'```


# Step 1: Process datasets to produce inputs for LM

The datasets are in `datasets` folder. proc_datasets.py will take these datasets and convert into XXX-contextlist and XXX-targetlist files. For example - Role-88 dataset will be processed and saved as `role-contextlist` and `role-targetlist` in `output` folder.

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

After running this script, there will be eight files in the `processed_datasets` folder-

For ROLE88 - `role-contextlist` , `role-targetlist`, 

For NEG-88 -`negsimp-contextlist`, `negsimp-contextlist`,  `negnat-contextlist`, `negnat-contextlist`, 

For CPRAG-34 - `cprag-contextlist`, `cprag-contextlist`, 


# Step 2: Get LM predictions/probabilities

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

After running get_model_responses.py script, there will be eight files in the outputs folder here for examples its  `outputs/bert-base/`.Out of eight files four are for top k predictions and four files are probabilities.   

The output prediction files will be (for example for bert-base-uncased model)

        modelpreds-cprag-bert-base-uncased  
        modelpreds-negnat-bert-base-uncased 
        modelpreds-negsimp-bert-base-uncased 
        modelpreds-role-bert-base-uncased 

The output probabilities will be stored in 

        modeltgtprobs-cprag-bert-base-uncased
        modeltgtprobs-negnat-bert-base-uncased
        modeltgtprobs-negsimp-bert-base-uncased
        modeltgtprobs-role-bert-base-uncased

 

# Step 3: Run accuracy and sensitivity tests for each diagnostic

```
python prediction_accuracy_tests.py \
  --preddir <location of top k prediction and  probabilities output folder> \
  --resultsdir <location of the result folder> \
  --models <modelname> \
  --k_values <k value> \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv
  ```
for example

```
python prediction_accuracy_tests.py \
  --preddir outputs/bert-base/ \
  --resultsdir result-acc-sentivity/bert-base \
  --models bert-base-uncased \
  --k_values 5 \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv
  ```

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