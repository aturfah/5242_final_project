from misc import pull_data
from config import Config
from pprint import pprint

def count_list_elements(list_to_count):
    output = {}
    for elt in sorted(list_to_count):
        if elt not in output:
            output[elt] = 0
        
        output[elt] += 1

    return output

if __name__ == "__main__":
    for dataset in Config.DATASETS:
        print("\n\n", dataset)

        train, valid, test = pull_data(dataset)
        for idx in range(len(train)):
            print("\n\tFOLD #{}".format(idx))
            train_fold = train[idx]
            valid_fold = valid[idx]
            test_fold = test

            train_labels = [x.numpy() for x in train_fold.map(lambda x, y: y)]
            valid_labels = [x.numpy() for x in valid_fold.map(lambda x, y: y)]
            test_labels = [x.numpy() for x in test_fold.map(lambda x, y: y)]

            pprint(count_list_elements(train_labels))
            pprint(count_list_elements(valid_labels))
            pprint(count_list_elements(test_labels))
