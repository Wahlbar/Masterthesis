""" Code taken from the vast library https://github.com/Vastlab/vast"""
from torch.nn import functional as f
import torch
from vast import tools
import torchvision

""" EOS with weighting with 1"""
class EntropicOpensetLoss:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, unk_weight=1):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.unknowns_multiplier = unk_weight / self.class_count
        self.ones = tools.device(torch.ones(self.class_count)) * self.unknowns_multiplier
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        # print("unk_weight = ", unk_weight, "\n")

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        categorical_targets[unk_idx, :] = (
            self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # print("Categorial Targets: ")
        # print(categorical_targets)
        # print("\n")
        # print("unk_idx: ")
        # print(unk_idx)
        # print("\n")
        # print("target: ")
        # print(target)
        # print("\n")
        # print("kn_idx: ")
        # print(kn_idx)
        # print("\n")
        # print("logits: ")
        # print(logits)
        # print("\n")

        return self.cross_entropy(logits, categorical_targets)

""" EOS with weighting with w = average_samples_per_known_class/number_of_negative_samples (both from training data)"""
class EntropicOpensetLoss1():
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.ones(len(target))
        weights[unk_idx] = 28895/30/31794

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")
        
        return mean_cross_entropy
    
""" EOS with unknown weighting with 0.5"""
class EntropicOpensetLoss2:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.ones(len(target))
        weights[unk_idx] = 0.5

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]
        
        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )
        
        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")

        return mean_cross_entropy

""" EOS with unknown weighting 0.1"""
class EntropicOpensetLoss3:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.ones(len(target))
        weights[unk_idx] = 0.1

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")
        
        return mean_cross_entropy
    
""" EOS with Softmax with background weighting: w_c = N/(C * N_c)"""
class EntropicOpensetLoss4:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, kn_w, ukn_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w
        self.unknown_weight = ukn_w        

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.zeros(len(target))

        #does .groupby() maintain the order?
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weights[i] = self.known_weights[category]
            else:
                weights[i] = self.unknown_weight

        weights[unk_idx] = self.unknown_weight
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")

        return mean_cross_entropy

    """ EOS with Softmax with domain-adapted multitask loss function (MOON-Paper). Implement last. Is similar to EOS4"""
class EntropicOpensetLoss5:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, kn_w, ukn_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w
        self.unknown_weight = ukn_w        

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.zeros(len(target))

        #does .groupby() maintain the order? Under the assumption that groupby orders the amount by ascending class labels.
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weights[i] = self.known_weights[category]
            else:
                weights[i] = self.unknown_weight

        weights[unk_idx] = self.unknown_weight
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")

        return mean_cross_entropy
    
    """ EOS with Focal Loss 1 ()"""
class EntropicOpensetLossFCL1:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma        

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        # weights = torch.zeros(len(target))
        # torch.set_printoptions(threshold=10_000)
        # print(softmax)
        #does .groupby() maintain the order?
        # for i in range(len(target)):
        #     if target[i] > -1:
        #         category = target[i]
        #         weights[i] = self.known_weights[category]
        #     else:
        #         weights[i] = self.unknown_weight

        # weights[unk_idx] = self.unknown_weight
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        softmax = torch.softmax(logits, dim=1)
        # calculate unweighted cross entropy, then multiply it with the weights.

        # weighted_loss = torch.zeros(logits.size(0), device=logits.device)
        
        # # ugly but works!
        # for i in range(len(weighted_loss)):
        #     logit = logits[i]
        #     categorical_target = categorical_targets[i]
        #     softmax_element = softmax[i]
        #     if unk_idx[i] == True:
        #         argmax_y = torch.max(softmax_element)
        #         weight = (-1) ** (self.gamma - 1) * self.alpha * (self.probability_per_class - argmax_y) ** self.gamma
        #     else:
        #         index = (categorical_target == 1).nonzero(as_tuple=True)[0]
        #         weight = -self.alpha * (1 - softmax_element[index]) ** self.gamma
        #     # print(weight)
        #     weighted_loss[i] = weight * self.cross_entropy(logit, categorical_target)

        # torch.set_printoptions(threshold=10_000)

        weighted_loss = torch.zeros(logits.size(0), device=logits.device)
        argmax_y = torch.max(softmax[unk_idx], dim=1).values
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)

        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma
        weight_unknown = (-1) ** (self.gamma - 1) * self.alpha * (
            self.probability_per_class - argmax_y
        ) ** self.gamma
        weighted_loss[kn_idx] = weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )
        weighted_loss[unk_idx] = weight_unknown * self.cross_entropy(
            logits[unk_idx], categorical_targets[unk_idx]
        )




        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_loss)
        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")

        return mean_cross_entropy
    
    """ EOS with Focal Loss 2 with sum over all classes ()"""
class EntropicOpensetLossFCL2:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma         

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.zeros(len(target))

        #does .groupby() maintain the order?
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weights[i] = self.known_weights[category]
            else:
                weights[i] = self.unknown_weight

        weights[unk_idx] = self.unknown_weight
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")

        return mean_cross_entropy

    """ EOS with Focal Loss 3 with class weights ()"""
class EntropicOpensetLossFCL3:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha, kn_w, ukn_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w
        self.unknown_weight = ukn_w
        self.alpha = alpha
        self.gamma = gamma          

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        weights = torch.zeros(len(target))

        #does .groupby() maintain the order?
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weights[i] = self.known_weights[category]
            else:
                weights[i] = self.unknown_weight

        weights[unk_idx] = self.unknown_weight
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[unk_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        # print("Weights: ")
        # print(weights)
        # print("Categorial Targets: ")
        # torch.set_printoptions(threshold=10_000)
        # print(categorical_targets)
        # print("\n")
        # print("weighted_cross_entropy: ")
        # print(weighted_cross_entropy)
        # print("\n")
        # print("summed_cross_entropy: ")
        # print(summed_cross_entropy)
        # print("\n")

        return mean_cross_entropy

class AverageMeter(object):
    """ Computes and stores the average and current value. Taken from
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.reset()

    def reset(self):
        """ Sets all values to 0. """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        """ Update metric values.

        Args:
            val (flat): Current value.
            count (int): Number of samples represented by val. Defaults to 1.
        """
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:3.3f}"


# Taken from:
# https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks/
class EarlyStopping:
    """ Stops the training if validation loss/metrics doesn't improve after a given patience"""
    def __init__(self, patience=100, delta=0):
        """
        Args:
            patience(int): How long wait after last time validation loss improved. Default: 100
            delta(float): Minimum change in the monitored quantity to qualify as an improvement
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss is True:
            score = -metrics
        else:
            score = metrics

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
