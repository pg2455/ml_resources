# basic imports
import numpy as np
import matplotlib.pyplot as plt
import math
import pathlib
import random

import torch
import torch.nn as nn
from torch.nn.modules.transformer import MultiheadAttention
from torch.optim.lr_scheduler import LambdaLR


class PriorDataLoader(object):
    def __init__(self, get_prior_data_fn, batch_size, num_points_in_each_dataset):
        self.batch_size = batch_size
        self.num_points_in_each_dataset = num_points_in_each_dataset
        self.get_prior_data_fn = get_prior_data_fn
        
        self.epoch = 0
        
    def get_batch(self, train = True, batch_size=None):
        """
        Returns:
            xs, ys, trainset_size
        """
        self.epoch += train
        bs = batch_size if batch_size else self.batch_size
        return self.get_prior_data_fn(bs, self.num_points_in_each_dataset), self._sample_trainset_size()

    def _sample_trainset_size(self):
        # samples a number between 1 and n-1 with higher weights to larger numbers
        # Appendix E.1 of Muller et al. (2021)
        min_samples = 1
        max_samples = self.num_points_in_each_dataset - 1
        
        sampling_weights = [1 / (max_samples - min_samples - i) for i in range(max_samples - min_samples)]
        return random.choices(range(min_samples, max_samples), sampling_weights)[0]
    

def get_bucket_limts(num_outputs, ys):
    """
    Creates buckets based on the values in y. 
    
    Args:
        num_outputs: number of buckets to create
        ys: values of y in the prior
    
    Returns:
        bucket_limits: An array containing the borders for each bucket. 
    """
    ys  = ys.flatten()

    if len(ys) % num_outputs:
        ys = ys[:-(len(ys) % num_outputs)]

    ys_per_bucket = len(ys) // num_outputs
    full_range = (ys.min(), ys.max())

    ys_sorted, _ = ys.sort(0)

    bucket_limits = (ys_sorted[ys_per_bucket-1::ys_per_bucket][:-1] + ys_sorted[ys_per_bucket::ys_per_bucket]) / 2
    bucket_limits = torch.cat([full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)], dim=0)
    return bucket_limits


def y_to_bucket_idx(ys, bl):
    """
    Maps the value of y to the corresponding bucket in `bl`
    
    Args:
        ys: value of y to be mapped into a bucket in bl
        bl: bucket limits specifiying the borders of buckets
    
    Returns:
        values of corresponding bucket number for y in bl
    """
    target_sample = torch.searchsorted(bl, ys) - 1
    target_sample[ys <= bl[0]] = 0
    target_sample[ys >= bl[-1]] = len(bl) - 1 - 1
    return target_sample


class Encoder(nn.Module):
    """Typical self attention module in transformer"""
    def __init__(self, d_model, n_heads, n_hidden, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, d_model),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, src, trainset_size):
        src_left, src_right = src[:, :trainset_size], src[:, trainset_size:]
        x_left = self.self_attn(src_left, src_left, src_left)[0] # all train points to each other
        x_right = self.self_attn(src_right, src_left, src_left)[0] # test points attend to train points
        x = torch.cat([x_left, x_right], dim=1)
        x = self.norm1(src + self.dropout(x))
        return self.norm2(self.dropout(self.out(x)) + x)


class Transformer(nn.Module):
    def __init__(self, num_features, n_out, n_layers=2, d_model=512, n_heads=4, n_hidden=1024, dropout=0.0, normalize=lambda x:x):
        super().__init__()
        
        self.x_encoder = nn.Linear(num_features, d_model)
        self.y_encoder = nn.Linear(1, d_model)
        
        self.model = nn.ModuleList(
            [Encoder(d_model, n_heads, n_hidden, dropout) for _ in range(n_layers)]
        )
        
        self.out = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_out)
        )
                
        self.normalize = normalize
        self.init_weights()
        
    def forward(self, x, y, trainset_size):
        """
        Args:
            x: num_datasets x number_of_points x num_features
            y: num_datasets x number_of_points
            trainset_size: int specifying the number of points to use as training dataset size
        
        Returns:
            outputs for each x
        """
        x_src = self.x_encoder(self.normalize(x))
        y_src = self.y_encoder(y)
        
        src = torch.cat([x_src[:, :trainset_size] + y_src[:, :trainset_size], x_src[:, trainset_size:]], dim=1)
        for encoder in self.model:
            src = encoder(src, trainset_size)

        return self.out(src)
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                

def get_halfnormal_with_p_weight_before(range_max, p=0.5):
    """
    Constructs a half normal distribution. 
    Args:
        range_max: used for scaling the half normal so that `p` portion of the half normal lies within range_max
        p: Cumulative probability under half normal

    Returns:
        half normal distribution
    """
    s = range_max / torch.distributions.HalfNormal(torch.tensor(1.)).icdf(torch.tensor(p))
    return torch.distributions.HalfNormal(s)


def compute_bar_distribution_loss(logits, target_y, bucket_limits, label_smoothing=0.0):
    """
    Implements Reimann distribution for logits. See Appendix E of Muller et al. 2021.
    
    Args:
        logits: num_datasets  x num_points_in_each_dataset x num_outputs_for_classification
        target_y: target class for each point 
        bucket_limits: border limits for each class
        label_smoothing: constant to define the amount of label smoothing to be used.
    
    Returns:
        loss: scalar value
    
    """
    target_y_idx = y_to_bucket_idx(target_y, bucket_limits)

    bucket_widths = bucket_limits[1:] - bucket_limits[:-1]
    bucket_log_probs = torch.log_softmax(logits, -1)
    scaled_bucket_log_probs = bucket_log_probs - torch.log(bucket_widths) # Refer to the equation above
    log_probs = scaled_bucket_log_probs.gather(-1, target_y_idx[..., None]).squeeze(-1)
    
    # full support distribution correction using half normals
    side_normals = (
        get_halfnormal_with_p_weight_before(bucket_widths[0]),
        get_halfnormal_with_p_weight_before(bucket_widths[-1])
    )
    # Correction for the bucket in the starting
    first_bucket = target_y_idx == 0
    log_probs[first_bucket] += side_normals[0].log_prob((bucket_limits[1] - target_y[first_bucket])).clamp(min=1e-8) + torch.log(bucket_widths[0])

    # Correction for the bucket at the end
    last_bucket = target_y_idx == len(bucket_widths) - 1
    log_probs[last_bucket] += side_normals[1].log_prob((target_y[last_bucket] - bucket_limits[-2])).clamp(min=1e-8) + torch.log(bucket_widths[-1])
    
    nll_loss = -log_probs
    smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
    
    loss = (1-label_smoothing) * nll_loss + label_smoothing * smooth_loss
    return loss.mean()


# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def compute_mean(logits, bucket_limits):
    """
    Computes mean from logits
    
    Args:
        logits: num_datasets  x num_points_in_each_dataset x num_outputs_for_classification
        bucket_limits: border limits for each class
    
    Returns:
        means
    """
    bucket_widths = bucket_limits[1:] - bucket_limits[:-1]
    bucket_means = bucket_limits[:-1] + bucket_widths / 2
    p = torch.softmax(logits, dim=-1)
    
    # full support correction through half normal (see Appendix E of Muller et al. 2021)
    side_normals = (
        get_halfnormal_with_p_weight_before(bucket_widths[0]),
        get_halfnormal_with_p_weight_before(bucket_widths[-1])
    )
    bucket_means[0] = -side_normals[0].mean + bucket_widths[1]
    bucket_means[-1] = side_normals[1].mean + bucket_widths[-2]
    
    return p @ bucket_means


def cdf(logits, left_prob, bucket_limits):
    """
    Computes quantile corresponding to left_prob
    
    Args:
        logits: num_datasets  x num_points_in_each_dataset x num_outputs_for_classification
        left_prob: probability of the output to be less than this value
        bucket_limits: border limits for each class
    
    Returns:
        points corresponding to left_prob
    
    """
    probs = torch.softmax(logits, -1)
    cumprobs = torch.cumsum(probs, -1)
    idx = torch.searchsorted(cumprobs, left_prob * torch.ones(*cumprobs.shape[:-1], 1)).squeeze(-1).clamp(0, cumprobs.shape[-1] - 1)
    cumprobs = torch.cat([torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1)
    
    rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
    left_border = bucket_limits[idx]
    right_border = bucket_limits[idx+1]
    return left_border + (right_border - left_border) * rest_prob / probs.gather(-1, idx[..., None]).squeeze(-1)
    

def compute_quantile(logits, bucket_limits, center_prob = 0.682):
    """
    Computes +/- quantiles corresponding to cetner_prob.
    
    Args:
        logits: num_datasets  x num_points_in_each_dataset x num_outputs_for_classification
        bucket_limits: border limits for each class
        center_prob: Probability mass for the points to be in the middle of mean +/- quantile value
    
    Returns:
        left quantile, right quantile
    
    """
    # center_prob = 0.682 area under the normal function within 1 std
    side_prob = (1.0-center_prob)/2
    
    # compute the quantile corresp. to this side_prob and  1-side_prob
    return torch.stack([cdf(logits, side_prob, bucket_limits), cdf(logits, 1-side_prob, bucket_limits)], -1)
     

def train(get_prior_data, num_features, num_outputs, num_points_in_each_dataset, max_num_datasets, hyperparameters, train_parameters, model_parameters, device):
    """
    Args:
        get_prior_data: Function to generate prior data
        num_features: number of input features
        num_outputs: dimension of the model output (number of possible classes for the output)
        num_points_in_each_dataset: number of points in each of the dataset
        max_num_datasets: maximum number of datasets in a batch
        hyperparameters: hyperparameters for get_prior_data
        train_parameters: parameters for training
        model_parameters: parameters to define the model
        device: cpu or cuda; torch.device
    
    Return:
        trained model, dataset generator, bucket_limits
        
    """
    # Reimannian distribution (this might take time)
    calibrate_datasets = hyperparameters.get('calibrate_datasets', 100000)
    _, ys, _ = get_prior_data(calibrate_datasets, num_features, num_points_in_each_dataset, hyperparameters)
    bucket_limits = get_bucket_limts(num_outputs, ys)

    # training data
    get_prior_data_fn = lambda x, y: get_prior_data(x, num_features, y, hyperparameters)
    dl = PriorDataLoader(get_prior_data_fn, batch_size=max_num_datasets, num_points_in_each_dataset=num_points_in_each_dataset)

    # Models
    normalize = model_parameters['normalize']
    n_out = model_parameters['n_out']
    d_model = model_parameters['d_model']
    n_layers = model_parameters['n_layers']
    n_hidden = model_parameters['n_hidden']
    n_heads = model_parameters['n_heads']
    model = Transformer(num_features, n_out=n_out, d_model=d_model, n_layers=n_layers, n_hidden=n_hidden, n_heads=n_heads, normalize=normalize)

    # Learning
    epochs = train_parameters['epochs']
    steps_per_epoch = train_parameters['steps_per_epoch']
    warmup_epochs = epochs // 4
    validate_epoch = 10
    lr = 0.001

    model.to(device)
    bucket_limits = bucket_limits.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)

    # test data
    (test_xs, test_ys, test_target_ys), test_trainset_size = dl.get_batch(False, 64)
    test_xs, test_ys, test_target_ys = test_xs.to(device), test_ys.to(device), test_target_ys.to(device)

    for step in range(epochs * steps_per_epoch + 1):
        (xs, ys, target_ys), trainset_size = dl.get_batch()
        xs, ys, target_ys = xs.to(device), ys.to(device), target_ys.to(device)
        pred_y = model(xs, ys.unsqueeze(-1), trainset_size)

        logits = pred_y[:, trainset_size:]
        target_y = target_ys[:, trainset_size:].clone().view(*logits.shape[:-1])

        loss = compute_bar_distribution_loss(logits, target_y, bucket_limits)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()

        if step % (validate_epoch * steps_per_epoch) == 0:
            with torch.no_grad():
                pred_test_y = model(test_xs, test_ys.unsqueeze(-1), test_trainset_size)

                logits = pred_test_y[:, test_trainset_size:]
                target_test_y = test_target_ys[:, test_trainset_size:].clone().view(*logits.shape[:-1])

                val_loss = compute_bar_distribution_loss(logits, target_test_y, bucket_limits)

            print(f'@Step: {step}\tVal loss:{val_loss.item(): 0.5f}\t lr:{scheduler.get_last_lr()[0]: 0.6e}\t train_loss:{loss.item(): 0.5f}')

    return model, dl, bucket_limits
    
    
def visualize(model, dl, bucket_limits, device, n_train=None):
    """
    Plots model predictions for small data.
    
    Args:
        model: model to process x,y from dl
        dl: data loader
        bucket_limits: border limits for each class
        device: cpu or cuda; torch.device
    """
    (xs, ys, target_ys), num_training_points = dl.get_batch()

    batch_index = np.random.randint(xs.shape[0])
    num_training_points = 5 if n_train is None else n_train

    xs_train = xs[batch_index, :num_training_points].detach().cpu().numpy()
    ys_train = ys[batch_index, :num_training_points].detach().cpu().numpy()
    xs_test = xs[batch_index, num_training_points:].detach().cpu().numpy()

    with torch.no_grad():
        logits = model(xs.to(device), ys.unsqueeze(-1).to(device), num_training_points)

    # predicted means from a bar distribution
    test_logits = logits[batch_index, num_training_points:].cpu()
    pred_means = compute_mean(test_logits, bucket_limits.cpu())
    pred_confs = compute_quantile(test_logits, bucket_limits.cpu())

    # plot pfn
    plt.scatter(xs_train[..., 0], ys_train)
    order_test_x = xs_test[..., 0].argsort()
    plt.plot(xs_test[order_test_x, 0], pred_means[order_test_x], color='green', label='pfn')
    plt.fill_between(xs_test[order_test_x, 0], pred_confs[order_test_x, 0], pred_confs[order_test_x, 1], alpha=0.1, color='green')