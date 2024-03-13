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

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        categorical_targets[unk_idx, :] = (
            self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )

        return self.cross_entropy(logits, categorical_targets)

""" EOS with weighting for negative samples with w = average_samples_per_known_class/number_of_negative_samples (both from training data)"""
class EntropicOpensetLossBalancedNegative():
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, neg_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.negative_weight = neg_w  
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        neg_idx = target < 0
        kn_idx = ~neg_idx

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = self.cross_entropy(logits, categorical_targets)

        # multiplicate the negative indices with a weight of 
        weighted_cross_entropy[neg_idx] *= self.negative_weight

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        return mean_cross_entropy
    
""" EOS with negative weighting with 0.5"""
class EntropicOpensetLoss0_5:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        neg_idx = target < 0
        kn_idx = ~neg_idx

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]
        
        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )
        
        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = self.cross_entropy(logits, categorical_targets)

        # multiplicate the negative indices with a weight of 0.5
        weighted_cross_entropy[neg_idx] *= 0.5

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        return mean_cross_entropy

""" EOS with negative weighting 0.1"""
class EntropicOpensetLoss0_1:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count) * self.probability_per_class)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        neg_idx = target < 0
        kn_idx = ~neg_idx

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        
        
        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = self.cross_entropy(logits, categorical_targets)

        # multiplicate the negative indices with a weight of 0.1
        weighted_cross_entropy[neg_idx] *= 0.1

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)
        
        return mean_cross_entropy
    
""" EOS with Softmax with background weighting: w_c = N/(C * N_c)"""
class EntropicOpensetLossBalancedComplete:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, kn_w, neg_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w
        self.negative_weight = neg_w        

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        neg_idx = target < 0
        kn_idx = ~neg_idx
        weights = tools.device(torch.zeros(len(target)))

        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weights[i] = self.known_weights[category]
            else:
                weights[i] = self.negative_weight

        weights[neg_idx] = self.negative_weight

        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )
        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        return mean_cross_entropy
    
""" EOS filler to try out new stuff"""
class EntropicOpensetLossF:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, kn_w, neg_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w
        self.negative_weight = neg_w        

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        neg_idx = target < 0
        kn_idx = ~neg_idx
        weights = tools.device(torch.zeros(len(target)))

        #does .groupby() maintain the order?
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weights[i] = self.known_weights[category]
            else:
                weights[i] = self.negative_weight

        weights[neg_idx] = self.negative_weight

        # check if there is known samples in the bastch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )
        # calculate unweighted cross entropy, then multiply it with the weights.
        weighted_cross_entropy = weights * self.cross_entropy(logits, categorical_targets)

        # take the mean of the cross entropy
        mean_cross_entropy = torch.mean(weighted_cross_entropy)

        return mean_cross_entropy

    """ EOS with Focal Loss 1 ()"""
class EntropicOpensetFocalLossMax:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # for negative samples: get the highest probability of the known classes (for each sample)
        argmax_y = torch.max(softmax[neg_idx], dim=1).values

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)

        # define the weight of the known samples
        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma

        # define the weight of the negative samples
        weight_negative = -self.alpha * abs((self.probability_per_class - argmax_y)) ** self.gamma

        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        # calculate the weighted cross_entropy of the negative samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[neg_idx] = -weight_negative * self.cross_entropy(
            logits[neg_idx], categorical_targets[neg_idx]
        )

        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss
    
    """ EOS with Focal Loss 2 with sum over all classes ()"""
class EntropicOpensetFocalLossSum:
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
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)

        # define the weight of the known samples
        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma

        # calculate the weighted cross_entropy of the known samples
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        weight_negative = -self.alpha * abs((categorical_targets[neg_idx] - softmax[neg_idx])) ** self.gamma 

        weighted_loss[neg_idx] = (weight_negative * torch.log(softmax[neg_idx])).sum(dim=1)

        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)

        return mean_focal_loss
    
""" EOS with Focal Loss 3 with sum over all samples bigger than 1/c"""
class EntropicOpensetFocalLoss3:
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
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx
        threshold = self.probability_per_class

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # # This focal loss on known samples approach is probably not that useful...

        # weighted_loss2 = torch.zeros(logits.size(0), device=logits.device)
        
        # for i in range(len(weighted_loss2)):
        #     targ = target[i]
        #     categorical_target = categorical_targets[i]
        #     softmax_element = softmax[i]
        #     if kn_idx[i] == True:
        #         print("categorical targets", categorical_target)
        #         print("target", targ)
        #         print("softmax element", softmax_element)
        #         argmax_y = torch.max(softmax_element)
        #         idx_kn = torch.argmax(softmax_element)
        #         print("idx kn", idx_kn)
        #         print("arg_max y", argmax_y)
        #         print("categorical target", categorical_target[idx_kn])
        #         weight = (-1) ** (self.gamma - 1) * self.alpha * (categorical_target[idx_kn] - argmax_y) ** self.gamma    
        #         print("weight", weight)

        # # for positive samples: get the highest probability of any classes (for each sample)
        # y = torch.max(softmax[kn_idx], dim=1).values
        # idx = torch.argmax(softmax[kn_idx], dim=1)
        
        # # define the weight of the positive samples
        # weight_known = -self.alpha * abs((categorical_targets[kn_idx, idx] - y)) ** self.gamma
    
        # print("targets", categorical_targets[kn_idx, idx])
        # print("y", y)
        # print("idx", idx)
        # print("weights", weight_known)

        # # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        # weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
        #     logits[kn_idx], categorical_targets[kn_idx]
        # )

        # print(weighted_loss)

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)

        # define the weight of the known samples
        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma
        
        # calculate the weighted cross_entropy of the known samples       
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        for i, (softmax_element, is_neg) in enumerate(zip(softmax, neg_idx)):
            if is_neg:
                softmax_l = softmax_element > threshold
                cat_target = categorical_targets[i]
                weight = (cat_target[softmax_l] - softmax_element[softmax_l]) ** self.gamma
                weighted_loss[i] = (-1) ** (self.gamma - 1) * self.alpha * torch.sum(weight * torch.log(softmax_element[softmax_l]))
        
        # softmax_l2 = softmax[neg_idx] > threshold Maybe use for optimization
        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        print(weighted_loss)
        # TODO: here
        return mean_focal_loss
    
    """ EOS with Focal Loss Filler to try out new stuff ()"""
class EntropicOpensetFocalLossF:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        softmax = torch.softmax(logits, dim=1)

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # for negative samples: get the highest probability of the known classes (for each sample)
        argmax_y = torch.max(softmax[neg_idx], dim=1).values

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)

        # define the weight of the known samples
        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma

        # define the weight of the negative samples
        weight_negative = -self.alpha * abs((self.probability_per_class - argmax_y)) ** self.gamma

        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        # calculate the weighted cross_entropy of the negative samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[neg_idx] = -weight_negative * self.cross_entropy(
            logits[neg_idx], categorical_targets[neg_idx]
        )

        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss
    
    """ EOS with Focal Loss only for known samples. Weighting of negative samples according to softmax with BG."""
class EntropicOpensetFocalLossKnown:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha, neg_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.negative_weight = neg_w 

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)

        # define the weight of the known samples
        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma

        # define the negative weights
        weight_negative = self.negative_weight

        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        # calculate the weighted cross_entropy of the negative samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[neg_idx] = weight_negative * self.cross_entropy(
            logits[neg_idx], categorical_targets[neg_idx]
        )

        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss

    """ EOS with Focal Loss only for negative samples. Weighting of known samples according to softmax with BG. """
class EntropicOpensetFocalLossNegative:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha, kn_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.probability_per_class = 1 / self.class_count
        self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # generate the weight_known tensor
        weight_known = tools.device(torch.zeros(len(target))) 

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # for negative samples: get the highest probability of the known classes (for each sample)
        argmax_y = torch.max(softmax[neg_idx], dim=1).values

        # define the weight of the known samples
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weight_known[i] = self.known_weights[category]

        # define the weight of the negative samples
        weight_negative = -self.alpha * abs((self.probability_per_class - argmax_y)) ** self.gamma

        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[kn_idx] = weight_known[kn_idx] * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        # calculate the weighted cross_entropy of the negative samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[neg_idx] = -weight_negative * self.cross_entropy(
            logits[neg_idx], categorical_targets[neg_idx]
        )

        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss
    
    """ EOS with Focal Loss 3 with sum over all samples bigger than 1/c"""
class EntropicOpensetFocalLossNegative2:
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
        
        # generate a tensor for the categorical targets
        categorical_targets = tools.device(torch.zeros(logits.shape))

        # tensor with booleans for known and negative samples
        neg_idx = target < 0
        kn_idx = ~neg_idx
        threshold = self.probability_per_class

        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # generates a [1/c, 1/c, ...] vector with length c & replaces the categorical classes in it.
        categorical_targets[neg_idx, :] = (
            self.negative_probabilities.expand(
                torch.sum(neg_idx).item(), self.class_count
            )
        )

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # generate the weight_known tensor
        weight_known = tools.device(torch.zeros(len(target))) 

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)

        # define the weight of the known samples
        for i in range(len(target)):
            if target[i] > -1:
                category = target[i]
                weight_known[i] = self.known_weights[category]
        
        # calculate the weighted cross_entropy of the known samples       
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        for i, (softmax_element, is_neg) in enumerate(zip(softmax, neg_idx)):
            if is_neg:
                softmax_l = softmax_element > threshold
                cat_target = categorical_targets[i]
                weight = (cat_target[softmax_l] - softmax_element[softmax_l]) ** self.gamma
                weighted_loss[i] = (-1) ** (self.gamma - 1) * self.alpha * torch.sum(weight * torch.log(softmax_element[softmax_l]))
        
        # softmax_l2 = softmax[neg_idx] > threshold Maybe use for optimization
        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        print(weighted_loss)
        # TODO: Here
        return mean_focal_loss
    
""" Softmax with Focal Loss 1 ()"""
class BackgroundFocalLoss:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        # self.probability_per_class = 1 / self.class_count
        # self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = self.eye[target]

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # define the weight of the known samples
        weights = -self.alpha * (1 - softmax[torch.arange(len(target)), target]) ** self.gamma

        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss = -weights * self.cross_entropy(
            logits, categorical_targets
        )
        
        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss

        
""" Softmax with Focal Loss Filler to try out new stuff ()"""
class BackgroundFocalLossKnown:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha, neg_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        # self.probability_per_class = 1 / self.class_count
        # self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.negative_weight = neg_w 

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = self.eye[target]

        # tensor with booleans for known and negative samples
        neg_idx = target == (self.class_count-1)
        kn_idx = ~neg_idx

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[kn_idx, :], dim=1)
        
        # define the weight of the known samples
        weight_known = -self.alpha * (1 - softmax[kn_idx, predicted_prob]) ** self.gamma

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)
        
        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[kn_idx] = -weight_known * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )

        weighted_loss[neg_idx] = self.negative_weight * self.cross_entropy(
            logits[neg_idx], categorical_targets[neg_idx]
        )

        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss

""" Softmax with Focal Loss Filler to try out new stuff ()"""
class BackgroundFocalLossNegative:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha, kn_w):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        # self.probability_per_class = 1 / self.class_count
        # self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.known_weights = kn_w

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = self.eye[target]

        # tensor with booleans for known and negative samples
        neg_idx = target == (self.class_count-1)
        kn_idx = ~neg_idx

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # for positive samples: get the probability of the correct class (for each sample)
        predicted_prob = torch.argmax(categorical_targets[neg_idx, :], dim=1)
        # define the weight of the known samples
        weight_negative = -self.alpha * (1 - softmax[neg_idx, predicted_prob]) ** self.gamma

        # generate the weighted loss 
        weighted_loss = torch.zeros(logits.size(0), device=logits.device)
        
        # generate the weight_known tensor
        weight_known = tools.device(torch.zeros(len(target))) 

        # define the weight of the known samples
        for i in range(len(target)):
            if target[i] != (self.class_count-1):
                category = target[i]
                weight_known[i] = self.known_weights[category]
        
        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss[neg_idx] = -weight_negative * self.cross_entropy(
            logits[neg_idx], categorical_targets[neg_idx]
        )

        weighted_loss[kn_idx] = weight_known[kn_idx] * self.cross_entropy(
            logits[kn_idx], categorical_targets[kn_idx]
        )
        
        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss

""" Softmax with Focal Loss Filler to try out new stuff ()"""
class BackgroundFocalLossF:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, gamma, alpha):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        # self.probability_per_class = 1 / self.class_count
        # self.negative_probabilities = tools.device(torch.ones(self.class_count)) * self.probability_per_class
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        # alpha = weighting factor of focal loss
        self.alpha = alpha
        # gamma = weighting exponential of focal loss
        self.gamma = gamma        

    def __call__(self, logits, target):
        
        # generate a tensor for the categorical targets
        categorical_targets = self.eye[target]

        # calculate the softmax (prediction probabilities) without gradients to no interfere in the calculation
        with torch.no_grad():
            softmax = torch.softmax(logits, dim=1)

        # define the weight of the known samples
        weights = -self.alpha * (1 - softmax[torch.arange(len(target)), target]) ** self.gamma

        # calculate the weighted cross_entropy of the known samples, minus because cross entopy itself already uses a minus in it. we have to revert it.
        weighted_loss = -weights * self.cross_entropy(
            logits, categorical_targets
        )
        
        # take the mean of the cross entropy
        mean_focal_loss = torch.mean(weighted_loss)
        return mean_focal_loss

    
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
