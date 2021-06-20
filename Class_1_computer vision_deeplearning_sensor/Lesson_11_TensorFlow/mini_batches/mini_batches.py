import math
import numpy as np
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    num_batches = len(features)//batch_size
    output = []
    for iter in range(0, num_batches):
        temp_feat = features[iter:((iter+1)*batch_size)]
        temp_label = labels[iter:((iter+1)*batch_size)]
        output.append([temp_feat,temp_label])
        print(iter)
        print(((iter+1)*batch_size))

    output.append([features[(num_batches*batch_size-1):len(features)], \
                   labels[(num_batches*batch_size-1):len(features)]])
    return output


features = np.zeros((5500, 16))
labels = np.zeros((5500, 8))
batch_size = 128

batch_result = batches(batch_size, features, labels)
batch_result = np.array(batch_result)

print(batch_result.shape)