import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generated-data", default=False, type=bool)
parser.add_argument("--testlist", default=None, type=str)

args = parser.parse_args()

if args.generated_data:
    
    os.system('python run-for-generated-data.py bert-base-uncased {}'.format(args.testlist))
    os.system('python run-for-generated-data.py bert-large-uncased {}'.format(args.testlist))

    os.system('python run-for-generated-data.py roberta-base {}'.format(args.testlist))
    os.system('python run-for-generated-data.py roberta-large {}'.format(args.testlist))

    os.system('python run-for-generated-data.py distilbert-base-uncased {}'.format(args.testlist))

    os.system('python run-for-generated-data.py albert-base-v1 {}'.format(args.testlist))
    os.system('python run-for-generated-data.py albert-large-v1 {}'.format(args.testlist))
    os.system('python run-for-generated-data.py albert-xlarge-v1 {}'.format(args.testlist))
    os.system('python run-for-generated-data.py albert-xxlarge-v1 {}'.format(args.testlist))

    os.system('python run-for-generated-data.py albert-base-v2 {}'.format(args.testlist))
    os.system('python run-for-generated-data.py albert-large-v2 {}'.format(args.testlist))
    os.system('python run-for-generated-data.py albert-xlarge-v2  {}'.format(args.testlist))
    os.system('python run-for-generated-data.py albert-xxlarge-v2  {}'.format(args.testlist))

    os.system('python run-for-generated-data.py t5-small {}'.format(args.testlist))
    os.system('python run-for-generated-data.py t5-base {}'.format(args.testlist))
    os.system('python run-for-generated-data.py t5-large {}'.format(args.testlist))
    os.system('python run-for-generated-data.py t5-3b {}'.format(args.testlist))

    os.system('python run-for-generated-data.py gpt2 {}'.format(args.testlist))
    os.system('python run-for-generated-data.py gpt2-medium {}'.format(args.testlist))
    os.system('python run-for-generated-data.py gpt2-large {}'.format(args.testlist))
    os.system('python run-for-generated-data.py gpt2-xl {}'.format(args.testlist))

  

else:
    
    os.system('python run.py bert-base-uncased')
    os.system('python run.py bert-large-uncased')
    
    os.system('python run.py roberta-base')
    os.system('python run.py roberta-large')

    os.system('python run.py distilbert-base-uncased')

    os.system('python run.py albert-base-v1')
    os.system('python run.py albert-large-v1')
    os.system('python run.py albert-xlarge-v1')
    os.system('python run.py albert-xxlarge-v1')

    os.system('python run.py albert-base-v2')
    os.system('python run.py albert-large-v2')
    os.system('python run.py albert-xlarge-v2')
    os.system('python run.py albert-xxlarge-v2')

    os.system('python run.py t5-small')
    os.system('python run.py t5-base')
    os.system('python run.py t5-large')
    os.system('python run.py t5-3b')

    os.system('python run.py gpt2')
    os.system('python run.py gpt2-medium')
    os.system('python run.py gpt2-large')
    os.system('python run.py gpt2-xl')