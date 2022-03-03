
# python prediction-accuracy-generated-data.py bert-base-uncased "negsimp"
import argparse
import os

def get_arguments():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("modelname", default='bert-base-uncased', help="huggingface model name")
    parser.add_argument("dataset", default='negsimp', help="'negsimp' for neg-simp, 'role' for role-88")
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    pred_file = "outputs/generated_data/{1}/modelpreds-{0}-{1}".format(args.dataset, args.modelname)
    tgt_file = "processed_datasets/generated_data/{}-targetlist".format(args.dataset)

    with open(pred_file) as f1, open(tgt_file) as f2:
        pred = f1.readlines()
        tgt = f2.readlines()
        num_correct = 0
        flipped_pred = 0
        top1correct = 0
        for i in range(0, len(tgt), 2):
            prediction = pred[i].strip()
            prediction = prediction.split(' ')
            if tgt[i].strip() in prediction:    # Number of correct predictions for affirmative sentences
                num_correct += 1
                if tgt[i].strip() == prediction[0]: # flipped_pred will track if the prediction (top 1) will change when moved from affirmative to negated sentences
                    top1correct += 1
                    if pred[i][0] != pred[i+1][0]:
                        flipped_pred += 1

    accuracy = num_correct / len(tgt_file)
    top1correct = top1correct / len(tgt_file) # number of correct in top 1 predictions
    sentitivity_neg = flipped_pred * 2 / len(tgt_file) # tgtfile has both affirmative and negated sentences, length will be half

    if not os.path.exists("result-acc-sentivity/generated_data/"):
        os.makedirs("result-acc-sentivity/generated_data/")

    textfile = open("result-acc-sentivity/generated_data/prediction-accuracy-{}".format(args.dataset), "a")
    textfile.write("Accuracy for {} model on {} is ".format(args.modelname, args.dataset) + str(accuracy) + "\n")
    textfile.close()

    textfile = open("result-acc-sentivity/generated_data/sensitivity-{}".format(args.dataset), "a")
    textfile.write("Top 1 correct predictions accuracy for {} model on {} = ".format(args.modelname, args.dataset) + str(top1correct) + " and Sensitivity to negation = " + str(sentitivity_neg) + "\n")
    textfile.close()

if __name__ == "__main__":
    main()