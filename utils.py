from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
from torch import nn, optim                             # nn for neural network layers and optim for training optimizers
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd                                     # Pandas to handle the data in dataframes
from datetime import datetime                           # datetime to use proper date and time formats
import os                                               # os handles directory/workspace changes
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm.auto import tqdm                              # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric
from sklearn.metrics import roc_auc_score               # ROC AUC model performance metric
import warnings                                         # Print warnings for bad practices
import sys                                              # Identify types of exceptions

# [TODO] Make the random seed a user option (randomly generated or user defined)
# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
# Automatic seed
# random_seed = np.random.get_state()
# np.random.set_state(random_seed)
# torch.manual_seed(random_seed[1][0])
# Manual seed
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Exceptions

class ColumnNotFoundError(Exception):
   """Raised when the column name is not found in the dataframe."""
   pass


# Auxiliary functions

def dataframe_to_padded_tensor(df, seq_len_dict, n_ids, n_inputs, id_column='subject_id', data_type='PyTorch', padding_value=999999):
    '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

    Parameters
    ----------
    df : pandas.Dataframe
        Data in a Pandas dataframe format which will be padded and converted
        to the requested data type.
    seq_len_dict : dictionary
        Dictionary containing the original sequence lengths of the dataframe.
    n_ids : int
        Total number of subject identifiers in a dataframe.
        Example: Total number of patients in a health dataset.
    n_inputs : int
        Total number of input features present in the dataframe.
    id_column : string, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    data_type : string, default 'PyTorch'
        Indication of what kind of output data type is desired. In case it's
        set as 'NumPy', the function outputs a NumPy array. If it's 'PyTorch',
        the function outputs a PyTorch tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    arr : torch.Tensor or numpy.ndarray
        PyTorch tensor or NumPy array version of the dataframe, after being
        padded with the specified padding value to have a fixed sequence
        length.
    '''
    # Max sequence length (e.g. patient with the most temporal events)
    max_seq_len = seq_len_dict[max(seq_len_dict, key=seq_len_dict.get)]

    # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
    arr = np.ones((n_ids, max_seq_len, n_inputs)) * padding_value

    # Iterator that outputs each unique identifier (e.g. each patient in the dataset)
    id_iter = iter(df[id_column].unique())

    # Count the iterations of ids
    count = 0

    # Assign each value from the dataframe to the numpy array
    for idt in id_iter:
        arr[count, :seq_len_dict[idt], :] = df[df[id_column] == idt].to_numpy()
        arr[count, seq_len_dict[idt]:, :] = padding_value
        count += 1

    # Make sure that the data type asked for is a string
    if not isinstance(data_type, str):
        raise Exception('ERROR: Please provide the desirable data type in a string format.')

    if data_type.lower() == 'numpy':
        return arr
    elif data_type.lower() == 'pytorch':
        return torch.from_numpy(arr)
    else:
        raise Exception('ERROR: Unavailable data type. Please choose either NumPy or PyTorch.')


def sort_by_seq_len(data, seq_len_dict, labels=None, id_column=0):
    '''Sort the data by sequence length in order to correctly apply it to a
    PyTorch neural network.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor on which sorting by sequence length will be applied.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    labels : torch.Tensor, default None
        Labels corresponding to the data used, either specified in the input
        or all the data that the interpreter has.
    id_column : int, default 0
        Number of the column which corresponds to the subject identifier in
        the data tensor.

    Returns
    -------
    sorted_data : torch.Tensor, default None
        Data tensor already sorted by sequence length.
    sorted_labels : torch.Tensor, default None
        Labels tensor already sorted by sequence length. Only outputed if the
        labels data is specified in the input.
    x_lengths : list of int
        Sorted list of sequence lengths, relative to the input data.
    '''
    # Get the original lengths of the sequences, for the input data
    x_lengths = [seq_len_dict[id] for id in list(data[:, 0, id_column].numpy())]

    is_sorted = all(x_lengths[i] >= x_lengths[i+1] for i in range(len(x_lengths)-1))

    if is_sorted:
        # Do nothing if it's already sorted
        sorted_data = data
        sorted_labels = labels
    else:
        # Sorted indeces to get the data sorted by sequence length
        data_sorted_idx = list(np.argsort(x_lengths)[::-1])

        # Sort the x_lengths array by descending sequence length
        x_lengths = [x_lengths[idx] for idx in data_sorted_idx]

        # Sort the data by descending sequence length
        sorted_data = data[data_sorted_idx, :, :]

        if labels is not None:
            # Sort the labels by descending sequence length
            sorted_labels = labels[data_sorted_idx, :]

    if labels is None:
        return sorted_data, x_lengths
    else:
        return sorted_data, sorted_labels,  x_lengths


def in_ipynb():
    '''Detect if code is running in a IPython notebook, such as in Jupyter Lab.'''
    try:
        return str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
    except:
        # Not on IPython if get_ipython fails
        return False


def pad_list(x_list, length, padding_value=999999):
    '''Pad a list with a specific padding value until the desired length is
    met.

    Parameters
    ----------
    x_list : list
        List which will be padded.
    length : int
        Desired length for the final padded list.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    x_list : list
        Resulting padded list'''
    return x_list + [padding_value] * (length - len(x_list))


def set_bar_color(values, ids, seq_len, threshold=0,
                  neg_color='rgba(30,136,229,1)', pos_color='rgba(255,13,87,1)'):
    '''Determine each bar's color in a bar chart, according to the values being
    plotted and the predefined threshold.

    Parameters
    ----------
    values : numpy.Array
        Array containing the values to be plotted.
    ids : int or list of ints
        ID or list of ID's that select which time series / sequences to use in
        the color selection.
    seq_len : int or list of ints
        Single or multiple sequence lengths, which represent the true, unpadded
        size of the input sequences.
    threshold : int or float, default 0
        Value to use as a threshold in the plot's color selection. In other
        words, values that exceed this threshold will have one color while the
        remaining have a different one, as specified in the parameters.
    pos_color : string
        Color to use in the bars corresponding to threshold exceeding values.
    neg_color : string
        Color to use in the bars corresponding to values bellow the threshold.

    Returns
    -------
    colors : list of strings
        Resulting bar colors list.'''
    if type(ids) is list:
        # Create a list of lists, with the colors for each sequences' instances
        return [[pos_color if val > 0 else neg_color for val in values[id, :seq_len]]
                for id in ids]
    else:
        # Create a single list, with the colors for the sequence's instances
        return [pos_color if val > 0 else neg_color for val in values[ids, :seq_len]]


def change_grad(grad, data, min=0, max=1):
    '''Restrict the gradients to only have valid values.

    Parameters
    ----------
    grad : torch.Tensor
        PyTorch tensor containing the gradients of the data being optimized.
    data : torch.Tensor
        PyTorch tensor containing the data being optimized.
    min : int, default 0
        Minimum valid data value.
    max : int, default 0
        Maximum valid data value.

    Returns
    -------
    grad : torch.Tensor
        PyTorch tensor containing the corrected gradients of the data being
        optimized.
    '''
    # Minimum accepted gradient value to be considered
    min_grad_val = 0.001

    for i in range(data.shape[0]):
        if (data[i] == min and grad[i] < 0) or (data[i] == max and grad[i] > 0):
            # Stop the gradient from excedding the limit
            grad[i] = 0
        elif data[i] == min and grad[i] > min_grad_val:
            # Make the gradient have a integer value
            grad[i] = 1
        elif data[i] == max and grad[i] < -min_grad_val:
            # Make the gradient have a integer value
            grad[i] = -1
        else:
            # Avoid any insignificant gradient
            grad[i] = 0

    return grad


def ts_tensor_to_np_matrix(data, feat_num=None, padding_value=999999):
    '''Convert a 3D PyTorch tensor, such as one representing multiple time series
    data, into a 2D NumPy matrix. Can be useful for applying the SHAP Kernel
    Explainer.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the three dimensional data being converted.
    feat_num : list of int, default None
        List of the column numbers that represent the features. If not specified,
        all columns will be used.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    data_matrix : numpy.ndarray
        NumPy two dimensional matrix obtained from the data after conversion.
    '''
    # View as a single sequence, i.e. like a dataframe without grouping by id
    data_matrix = data.contiguous().view(-1, data.shape[2]).detach().numpy()
    # Remove rows that are filled with padding values
    if feat_num is not None:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix[:, feat_num]]]
    else:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix]]
    return data_matrix


def model_inference(model, seq_len_dict, dataloader=None, data=None, metrics=['loss', 'accuracy', 'AUC'],
                    padding_value=999999, output_rounded=False, experiment=None, set_name='test',
                    seq_final_outputs=False, cols_to_remove=[0, 1]):
    '''Do inference on specified data using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which does the inference on the data.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches during inference.
    data : tuple of torch.Tensor, default None
        If a data loader isn't specified, the user can input directly a
        tuple of PyTorch tensor on which inference will be done. The first
        tensor must correspond to the features tensor whe second one
        should be the labels tensor.
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (loss), accuracy, AUC
        (Receiver Operating Curve Area Under the Curve), precision, recall
        and F1.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    output_rounded : bool, default False
        If True, the output is rounded, to represent the class assigned by
        the model, instead of just probabilities (>= 0.5 rounded to 1,
        otherwise it's 0)
    experiment : comet_ml.Experiment, default None
        Represents a connection to a Comet.ml experiment to which the
        metrics performance is uploaded, if specified.
    set_name : str
        Defines what name to give to the set when uploading the metrics
        values to the specified Comet.ml experiment.
    seq_final_outputs : bool, default False
        If set to true, the function only returns the ouputs given at each
        sequence's end.
    cols_to_remove : list of ints, default [0, 1]
        List of indeces of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as subject_id
        and ts (timestamp).

    Returns
    -------
    output : torch.Tensor
        Contains the output scores (or classes, if output_rounded is set to
        True) for all of the input data.
    metrics_vals : dict of floats
        Dictionary containing the calculated performance on each of the
        specified metrics.
    '''
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()

    # Create an empty dictionary with all the possible metrics
    metrics_vals = {'loss': None,
                    'accuracy': None,
                    'AUC': None,
                    'precision': None,
                    'recall': None,
                    'F1': None}

    # Initialize the metrics
    if 'loss' in metrics:
        loss = 0
    if 'accuracy' in metrics:
        acc = 0
    if 'AUC' in metrics:
        auc = 0
    if 'precision' in metrics:
        prec = 0
    if 'recall' in metrics:
        rcl = 0
    if 'F1' in metrics:
        f1_score = 0

    # Check if the user wants to do inference directly on a PyTorch tensor
    if dataloader is None and data is not None:
        features, labels = data[0].float(), data[1].float()             # Make the data have type float instead of double, as it would cause problems
        features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

        # Remove unwanted columns from the data
        features_idx = list(range(features.shape[2]))
        [features_idx.remove(column) for column in cols_to_remove]
        features = features[:, :, features_idx]
        scores = model.forward(features, x_lengths)                     # Feedforward the data through the model

        # Adjust the labels so that it gets the exact same shape as the predictions
        # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
        labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

        mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
        unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
        unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
        pred = torch.round(unpadded_scores)                             # Get the predictions

        if output_rounded:
            # Get the predicted classes
            output = pred.int()
        else:
            # Get the model scores (class probabilities)
            output = unpadded_scores

        if seq_final_outputs:
            # Only get the outputs retrieved at the sequences' end
            # Cumulative sequence lengths
            final_seq_idx = np.cumsum(x_lengths) - 1

            # Get the outputs of the last instances of each sequence
            output = output[final_seq_idx]

        if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
            # Calculate the number of true positives, false negatives, true negatives and false positives
            true_pos = int(sum(torch.masked_select(pred, unpadded_labels.byte())))
            false_neg = int(sum(torch.masked_select(pred == 0, unpadded_labels.byte())))
            true_neg = int(sum(torch.masked_select(pred == 0, (unpadded_labels == 0).byte())))
            false_pos = int(sum(torch.masked_select(pred, (unpadded_labels == 0).byte())))

        if 'loss' in metrics:
            metrics_vals['loss'] = model.loss(scores, labels, x_lengths).item() # Add the loss of the current batch
        if 'accuracy' in metrics:
            correct_pred = pred == unpadded_labels                          # Get the correct predictions
            metrics_vals['accuracy'] = torch.mean(correct_pred.type(torch.FloatTensor)).item() # Add the accuracy of the current batch, ignoring all padding values
        if 'AUC' in metrics:
            metrics_vals['AUC'] = roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the ROC AUC of the current batch
        if 'precision' in metrics:
            curr_prec = true_pos / (true_pos + false_pos)
            metrics_vals['precision'] = curr_prec                           # Add the precision of the current batch
        if 'recall' in metrics:
            curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['recall'] = curr_rcl                               # Add the recall of the current batch
        if 'F1' in metrics:
            # Check if precision has not yet been calculated
            if 'curr_prec' not in locals():
                curr_prec = true_pos / (true_pos + false_pos)
            # Check if recall has not yet been calculated
            if 'curr_rcl' not in locals():
                curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['F1'] = 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl) # Add the F1 score of the current batch

        return output, metrics_vals

    # Initialize the output
    output = torch.tensor([]).int()

    # Evaluate the model on the set
    for features, labels in dataloader:
        # Turn off gradients, saves memory and computations
        with torch.no_grad():
            features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
            features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

            # Remove unwanted columns from the data
            features_idx = list(range(features.shape[2]))
            [features_idx.remove(column) for column in cols_to_remove]
            features = features[:, :, features_idx]
            scores = model.forward(features, x_lengths)                     # Feedforward the data through the model

            # Adjust the labels so that it gets the exact same shape as the predictions
            # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
            labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
            labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

            mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
            unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
            unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
            pred = torch.round(unpadded_scores)                             # Get the predictions

            if output_rounded:
                # Get the predicted classes
                output = torch.cat([output, pred.int()])
            else:
                # Get the model scores (class probabilities)
                output = torch.cat([output.float(), unpadded_scores])

            if seq_final_outputs:
                # Indeces at the end of each sequence
                final_seq_idx = [n_subject*features.shape[1]+x_lengths[n_subject]-1 for n_subject in range(features.shape[0])]

                # Get the outputs of the last instances of each sequence
                output = output[final_seq_idx]

            if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
                # Calculate the number of true positives, false negatives, true negatives and false positives
                true_pos = int(sum(torch.masked_select(pred, unpadded_labels.byte())))
                false_neg = int(sum(torch.masked_select(pred == 0, unpadded_labels.byte())))
                true_neg = int(sum(torch.masked_select(pred == 0, (unpadded_labels == 0).byte())))
                false_pos = int(sum(torch.masked_select(pred, (unpadded_labels == 0).byte())))

            if 'loss' in metrics:
                loss += model.loss(scores, labels, x_lengths)               # Add the loss of the current batch
            if 'accuracy' in metrics:
                correct_pred = pred == unpadded_labels                      # Get the correct predictions
                acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the accuracy of the current batch, ignoring all padding values
            if 'AUC' in metrics:
                auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the ROC AUC of the current batch
            if 'precision' in metrics:
                curr_prec = true_pos / (true_pos + false_pos)
                prec += curr_prec                                           # Add the precision of the current batch
            if 'recall' in metrics:
                curr_rcl = true_pos / (true_pos + false_neg)
                rcl += curr_rcl                                             # Add the recall of the current batch
            if 'F1' in metrics:
                # Check if precision has not yet been calculated
                if 'curr_prec' not in locals():
                    curr_prec = true_pos / (true_pos + false_pos)
                # Check if recall has not yet been calculated
                if 'curr_rcl' not in locals():
                    curr_rcl = true_pos / (true_pos + false_neg)
                f1_score += 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl) # Add the F1 score of the current batch

    # Calculate the average of the metrics over the batches
    if 'loss' in metrics:
        metrics_vals['loss'] = loss / len(dataloader)
        metrics_vals['loss'] = metrics_vals['loss'].item()                  # Get just the value, not a tensor
    if 'accuracy' in metrics:
        metrics_vals['accuracy'] = acc / len(dataloader)
        metrics_vals['accuracy'] = metrics_vals['accuracy'].item()          # Get just the value, not a tensor
    if 'AUC' in metrics:
        metrics_vals['AUC'] = auc / len(dataloader)
    if 'precision' in metrics:
        metrics_vals['precision'] = prec / len(dataloader)
    if 'recall' in metrics:
        metrics_vals['recall'] = rcl / len(dataloader)
    if 'F1' in metrics:
        metrics_vals['F1'] = f1_score / len(dataloader)

    if experiment is not None:
        # Log metrics to Comet.ml
        if 'loss' in metrics:
            experiment.log_metric(f'{set_name}_loss', metrics_vals['loss'])
        if 'accuracy' in metrics:
            experiment.log_metric(f'{set_name}_acc', metrics_vals['accuracy'])
        if 'AUC' in metrics:
            experiment.log_metric(f'{set_name}_auc', metrics_vals['AUC'])
        if 'precision' in metrics:
            experiment.log_metric(f'{set_name}_prec', metrics_vals['precision'])
        if 'recall' in metrics:
            experiment.log_metric(f'{set_name}_rcl', metrics_vals['recall'])
        if 'F1' in metrics:
            experiment.log_metric(f'{set_name}_f1_score', metrics_vals['F1'])

    return output, metrics_vals
