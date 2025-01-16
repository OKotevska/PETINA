import math
import random
import torch
import numpy as np
from scipy import stats as st

# This class has the baseline models

def type_checking_and_return_lists(domain):
    if isinstance(domain, torch.Tensor):
        items, shape = torch_to_list(domain)
    elif isinstance(domain, np.ndarray):
        items, shape = numpy_to_list(domain)
    elif isinstance(domain, list):
        items = domain
        shape = 0 # no use
    else:
        raise ValueError("only takes list, ndarray, tensor type")
    
    return items, shape

def type_checking_return_actual_dtype(domain,result, shape):
    if isinstance(domain, torch.Tensor):
        return list_to_torch(result, shape)
    elif isinstance(domain, np.ndarray):
        return list_to_numpy(result, shape)
    else:  # list type
         return result
    

def applyFlipCoin(probability, domain):
    """
    Applies a "flip coin" mechanism to each item in a list based on a given probability.

    Parameters:
        probability (float): Probability (between 0 and 1) of returning `True` for each item.
        items (list): List of items to which the flip coin mechanism will be applied.

    Returns:
        list of bool: A list indicating `True` or `False` for each item in `items` based on the probability.
    """

    
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    
    items, shape = type_checking_and_return_lists(domain)
    
   
    prob = [np.random.rand() < probability for _ in items]

    result = []
    item_min = min(items)
    item_max = max(items)
    for p, n in zip(prob, items):
        if p == True:
            result.append(n)
        else:
            result.append(random.randint(item_min, item_max))

    return type_checking_return_actual_dtype(domain,result, shape)

    
   
    

def applyDPGaussian(domain, delta=10e-5, epsilon=1, gamma=1):
    """
    Applies Gaussian noise to the input data to achieve differential privacy.

    Parameters:
        data (np.ndarray): A NumPy array of values to which noise will be added.
        delta (float): Failure probability, typically a small value (default: 1e-5).
        epsilon (float): Privacy parameter, controls the level of noise added (default: 1.0).
        gamma (float): privacy parameter scaling (lower value means less noise)

    Returns:
        np.ndarray: The data with added Gaussian noise for differential privacy.
    """
    
    data, shape = type_checking_and_return_lists(domain)

    sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
    privatized = data + np.random.normal(loc=0, scale=sigma, size=len(data))

    return type_checking_return_actual_dtype(domain, privatized, shape)

    

def applyDPExponential(domain, sensitivity=1, epsilon=1, gamma=1.0):
    """
    A function that returns values with exponential Noise'

    Input:
        value: A list of values
        sensitivity: maximum amount by which a single individual's data can influence the output of a function. (default=1)
        epsilon: Privacy Parameter (default=1)
        gamma: privacy parameter scaling (lower value means less noise)
    Returns:
        str: A privatized list of values
    """
    data, shape = type_checking_and_return_lists(domain)

    # Calculate the scale of the exponential distribution
    scale = sensitivity * gamma / epsilon

    # Generate exponential noise (symmetric around 0)
    noise = np.random.exponential(scale, size=len(data))
    signs = np.random.choice([-1, 1], size=len(data))  # Randomly flip signs
    noise = noise * signs

    # Add noise to the original data
    privatized = np.array(data) + noise

    # convert back to list 
    privatized = privatized.tolist()

    return type_checking_return_actual_dtype(domain, privatized, shape)

def applyDPLaplace(domain, sensitivity=1, epsilon=1, gamma=1):
    """
    A function that returns values with laplace Noise'

    Input:
        value: A list of values
        sensitivity: maximum amount by which a single individual's data can influence the output of a function. (default=1)
        epsilon: Privacy Parameter (default=1)
        gamma (float): privacy parameter scaling (lower value means less noise)
    Returns:
        str: A privatized list of values
    """
    data, shape = type_checking_and_return_lists(domain)
    privatized = data + np.random.laplace(loc=0, scale=sensitivity*gamma/epsilon, size=len(data))

    return type_checking_return_actual_dtype(domain, privatized, shape)


def encode(response, domain):
    #print([1 if d == response else 0 for d in domain])
    return [1 if d == response else 0 for d in domain]


def perturb_bit(bit, p, q):
    # p = .75
    # q = .25

    sample = np.random.random()
    if bit == 1:
        if sample <= p:
            return 1
        else:
            return 0
    elif bit == 0:
        if sample <= q:
            return 1
        else:
            return 0

def perturb(encoded_response, p, q):
    return [perturb_bit(b, p, q) for b in encoded_response]

def aggregate(responses, p= .75, q= .25):
    # p = .75
    # q = .25

    sums = np.sum(responses, axis=0)
    n = len(responses)

    return [(v - n*q) / (p-q) for v in sums]

def unaryEncoding(value, p=0.75, q=0.25):
    """
    Applies unary encoding with differential privacy to a given domain, using random response perturbation 
    to create privatized responses based on specified probabilities `p` and `q`.

    Parameters:
        domain (list): A list of discrete values representing the input domain to be encoded and perturbed.
        p (float): The probability of keeping an encoded bit as `True` during perturbation. Default is 0.75.
        q (float): The probability of flipping an encoded bit to `True` when it is actually `False`. Default is 0.25.

    Returns:
        list: A list of counts for each value in the `domain`, representing the privatized responses 
              after applying unary encoding and perturbation.
    """

    domain, _ = type_checking_and_return_lists(value)
    unique_domain = list(set(domain))
   
    responses = [perturb(encode(r, unique_domain), p, q) for r in domain]   
    counts = aggregate(responses, p, q)
    t = list(zip(unique_domain, counts))     # t = pairwise original and perturbed count values

    return t

# compute q given p and epsilon
def get_q( p, eps):
    # Want p(1-q)/q(1-p) = exp(eps)
    # I.e q^{-1} -1 = (1-q)/q = exp(eps) * (1-p)/p
    qinv = 1 + (math.exp(eps) * (1.0-p)/ p)
    q = 1.0 / qinv
    return q

# Implementation from: https://github.com/apple/ml-projunit/blob/main/utilities.py 
# compute sigma for gaussian given
def get_gamma_sigma( p, eps):
    # Want p(1-q)/q(1-p) = exp(eps)
    # I.e q^{-1} -1 = (1-q)/q = exp(eps) * (1-p)/p
    qinv = 1 + (math.exp(eps) * (1.0-p)/ p)
    q = 1.0 / qinv
    gamma = st.norm.isf(q)
    # Now the expected dot product is (1-p)*E[N(0,1)|<gamma] + pE[N(0,1)|>gamma]
    # These conditional expectations are given by pdf(gamma)/cdf(gamma) and pdf(gamma)/sf(gamma)
    unnorm_mu = st.norm.pdf(gamma) * (-(1.0-p)/st.norm.cdf(gamma) + p/st.norm.sf(gamma))
    sigma = 1./unnorm_mu
    return gamma, sigma

# implementation from: https://github.com/apple/ml-projunit/blob/main/utilities.py#L194
def get_p( eps, return_sigma=False):
    # Mechanism:
    # With probability p, sample a Gaussian conditioned on g.x \geq gamma
    # With probability (1-p), sample conditioned on g.x \leq gamma
    # Scale g appropriately to get the expectation right
    # Let q(gamma) = Pr[g.x \geq gamma] = Pr[N(0,1) \geq gamma] = st.norm.sf(gamma)
    # Then density for x above threshold = p(x)  * p/q(gamma)
    # And density for x below threhsold = p(x) * (1-p)/(1-q(gamma))
    # Thus for a p, gamma is determined by the privacy constraint.
    plist = np.arange(0.01, 1.0, 0.01)
    glist = []
    slist = []
    for p in plist:
        gamma, sigma = get_gamma_sigma(p, eps)
        # thus we have to scale this rv by sigma to get it to be unbiased
        # The variance proxy is then d sigma^2
        slist.append(sigma)
        glist.append(gamma)
    ii = np.argmin(slist)
    if return_sigma:
        return plist[ii], slist[ii]
    else:
        return plist[ii]
    
# preserves epsilon-differential privacy (returns the exact value, not the noisy value)
# Sparse Vector technique (SVT) -- implementation from https://programming-dp.com/ch10.html 
def above_threshold_SVT( val, domain, T, epsilon):
    possible_val_list, shape = type_checking_and_return_lists(domain)
    T_hat = T + np.random.laplace(loc=0, scale = 2/epsilon)
    
    nu_i = np.random.laplace(loc=0, scale = 4/epsilon)
    if val + nu_i >= T_hat:
        return val
    # if the algorithm "fails", return a random val from possible val list
    # more convenient in certain use cases
    return random.choice(possible_val_list)
    


def she_perturb_bit( bit, epsilon = 0.1):
    return bit + np.random.laplace(loc=0, scale = 2 / epsilon)


def she_perturbation( encoded_response, epsilon = 0.1):
    return [she_perturb_bit(b, epsilon) for b in encoded_response]


def the_perturb_bit( bit, epsilon = 0.1, theta = 1.0):
    val = bit + np.random.laplace(loc=0, scale = 2 / epsilon)
    
    if val > theta:
        return 1.0
    else:
        return 0.0

def the_perturbation( encoded_response, epsilon = 0.1, theta = 1.0):
    return [the_perturb_bit(b, epsilon, theta) for b in encoded_response]


def the_aggregation_and_estimation( answers, epsilon = 0.1, theta = 1.0):
    p = 1 - 0.5 * pow(math.e, epsilon / 2 * (1.0 - theta))
    q = 0.5 * pow(math.e, epsilon / 2 * (0.0 - theta))
    
    sums = np.sum(answers, axis=0)
    n = len(answers)
    
    return [int((i - n * q) / (p-q)) for i in sums] 
    

# implementation from https://livebook.manning.com/book/privacy-preserving-machine-learning/chapter-4/v-4/103
def histogramEncoding(value):

    domain, shape = type_checking_and_return_lists(value)

    responses = [she_perturbation(encode(r, domain)) for r in domain]
    counts = aggregate(responses)
    t = list(zip(domain, counts))
    # print(t)
    privatized = []
    for i in range(len(t)):
        privatized.append(t[i][1])

        

    return type_checking_return_actual_dtype(value,privatized, shape)


def histogramEncoding_t( value):

    domain, shape = type_checking_and_return_lists(value)
    #responses = [she_perturbation(encode(r, domain)) for r in domain]
    #she_estimated_answers = np.sum([she_perturbation(encoding(r)) for r in adult_age], axis=0)
    the_perturbed_answers = [the_perturbation(encode(r, domain)) for r in domain]
    estimated_answers = the_aggregation_and_estimation(the_perturbed_answers)
    # counts = aggregate(responses)
    # t = list(zip(domain, counts))
    # privatized = []
    # for i in range(len(t)):
    #     privatized.append(t[i][1])

    return type_checking_return_actual_dtype(value, estimated_answers, shape)


# clipping
# tutorial on TensorFlow, Keras and PyTorch: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
def applyClipping( value, clipping):
    clipped = []
    for i in range(len(value)):
        if (value[i] >= clipping):
            clipped.append(clipping)
        else:
            clipped.append(value[i])

    return clipped


def applyClippingAdaptive(domain):
    """
    Applies adaptive clipping to the input data based on lower quantile value of 0.05.

    Parameters:
        domain (list, array-like or tensor): The dataset to process.
        
    Returns:
        list, array-like or tensor: The data with adaptive clipping applied.
    """
    value, shape = type_checking_and_return_lists(domain)
    
    lower_quantile = 0.05
    lower = np.quantile(value, lower_quantile)
    
    clipped_data = np.clip(value, lower, np.max(value))
    clipped_data = clipped_data.tolist()

    return type_checking_return_actual_dtype(domain, clipped_data, shape)


# clipping with DP - read the paper referenced above for more info
# check the funtionality code using this code https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3
def applyClippingDP( domain, clipping, sensitivity, epsilon):
    value, shape = type_checking_and_return_lists(domain)
    tmpValue = applyClipping(value, clipping)
    privatized = []
    for i in range(len(tmpValue)):
        privatized.append(tmpValue[i] + np.random.laplace(loc=0, scale = sensitivity/epsilon))
        

    return type_checking_return_actual_dtype(domain, privatized, shape)

# pruning
# implementation from here https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700324.pdf
def applyPruning( domain, prune_ratio):
    value, shape = type_checking_and_return_lists(domain)
    rnd_tmp = 1
    pruned = []
    for i in range(len(value)):
        if (abs(value[i]) < prune_ratio):
            rnd_tmp = random.random()
            if (abs(value[i]) > rnd_tmp * prune_ratio):
                if (value[i] > 0):
                    pruned.append(prune_ratio)
                else:
                    pruned.append(-prune_ratio)
            else:
                pruned.append(0)

    return  type_checking_return_actual_dtype(domain, pruned, shape)

def applyPruningAdaptive(domain):
    value, shape = type_checking_and_return_lists(domain)
    rnd_tmp = 1
    pruned = []
    prune_ratio = max(value) + 0.1
    for i in range(len(value)):
        if (abs(value[i]) < prune_ratio):
            rnd_tmp = random.random()
            if (abs(value[i]) > rnd_tmp * prune_ratio):
                if (value[i] > 0):
                    pruned.append(prune_ratio)
                else:
                    pruned.append(-prune_ratio)
            else:
                pruned.append(0)

    return type_checking_return_actual_dtype(domain, pruned, shape)

# prunning with DP - read the paper referenced above for more info
def applyPruningDP( domain, prune_ratio, sensitivity, epsilon):
    value, shape = type_checking_and_return_lists(domain)
    tmpValue = applyPruning(value, prune_ratio)
    privatized = []
    for i in range(len(tmpValue)):
        privatized.append(tmpValue[i] + np.random.laplace(loc=0, scale = sensitivity/epsilon))

    return type_checking_return_actual_dtype(domain, privatized, shape)

def unary_epsilon( p, q):
    return np.log((p*(1-q)) / ((1-p)*q))


def shuffle( a):
    """
    Simple
    a = [[1,2,3], [4,5,6], [7,8,9]]  # lists of 1D parameters from clients for shuffling
    result = shuffle(a) 
    # result =  [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
    """
    random.shuffle(a) 
    return a

def percentilePrivacy(domain, percentile):
    """
    Applies percentile privacy by setting values below the given percentile to zero.
    
    Parameters:
        data (list, array-like or tensor): The dataset to process.
        percentile (float): The lower percentile threshold (0-100).
        
    Returns:
        data: The data with values below the specified percentile to zero.
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")
    
    data, shape = type_checking_and_return_lists(domain)
    data = np.array(data)
    
    # Calculate the lower and upper bounds based on percentiles
    lower_bound = np.percentile(data, percentile)
    
    # Set values outside the bounds to zero
    data = np.where((data >= lower_bound), data, 0)
    
    data = data.tolist()

    return type_checking_return_actual_dtype(domain, data, shape)

def numpy_to_list(nd_array):
    flattened_list = nd_array.flatten().tolist()
    nd_array_shape = nd_array.shape

    return flattened_list, nd_array_shape

def list_to_numpy(flattened_list, nd_array_shape):
    reverted_ndarray = np.array(flattened_list).reshape(nd_array_shape)
    return reverted_ndarray

def torch_to_list(torch_tensor):
    flattened_list = torch_tensor.flatten().tolist()
    tensor_shape = torch_tensor.shape

    return flattened_list, tensor_shape

def list_to_torch(flattened_list, tensor_shape):
    reverted_tensor =  torch.as_tensor(flattened_list).reshape(tensor_shape)
    return reverted_tensor

