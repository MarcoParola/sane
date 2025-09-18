import torch
from torch import bincount, float as torch_float, argmax as torch_argmax
import os

class ClassificationMetrics():

    def __init__(self, num_classes: int, device: str, fromfile: str = None):
        self.num_classes = num_classes
        self.device = device
        if fromfile is not None and os.path.exists(f"{fromfile}.npy"): 
            self.load(fromfile)
            self.loaded = True
            return None

        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch_float, requires_grad=False)

    def __call__(self, y_pred, y_true):
        '''
            Updates the confusion matrix every time a new pair of predictions and labels is passed
        '''
        preds = y_pred.to(self.device); 
        lbls = y_true.to(self.device); 
        confusion_matrix = self.confusion_matrix.to(self.device)
        if isinstance(preds.tolist()[0], list): preds = torch_argmax(preds, dim=1)
        # bincount of the 1D flattening of CM, where lbls works as row counter (y_true on rows) and preds as column counter
        confusion_matrix += bincount(lbls*self.num_classes+preds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix = confusion_matrix.to("cpu")

    def __str__(self):
        '''
            Uses todict() dictionary of tuples where first element is the metric value, the second element is a boolean
            indicating if the metric must be converted into a percentage and generates a string of the retrieved dictionary
        '''
        return(repr({k: f"{f'{v[0]*100:.2f}%' if v[1] else f'{v[0]:.4f}'}" for k,v in self.todict().items()}))        
    
    def todict(self) -> dict[str, float]:
        return {
            'accuracy': (self.accuracy(), True), 'precision': (self.precision(), True), 
            'recall': (self.recall(), True), 'f1': (self.f1score(), True)
        }

    def accuracy(self):
        confusion_matrix = self.confusion_matrix.to(self.device)
        total_guesses = confusion_matrix.diag().sum()
        total_samples = confusion_matrix.sum(dim=1).sum()
        # If total_samples is zero, return 1.0 to avoid division by zero
        return torch.where(total_samples == 0, torch.tensor(1.0, device=self.device), total_guesses / total_samples).item()
    
    def recall(self):
        confusion_matrix = self.confusion_matrix.to(self.device)
        true_positives = confusion_matrix.diag()
        denominator = confusion_matrix.sum(dim=1)  # Sum over columns for each class
        # If there are no true positives and false negatives, return 1.0 to avoid division by zero
        return torch.where(denominator == 0, torch.tensor(1.0, device=self.device), true_positives / denominator).mean().item()
    
    def precision(self):
        confusion_matrix = self.confusion_matrix.to(self.device)
        true_positives = confusion_matrix.diag()
        denominator = confusion_matrix.sum(dim=0)  # Sum over rows for each class
        # If there are no true positives and false positives, return 1.0 to avoid division by zero
        return torch.where(denominator == 0, torch.tensor(1.0, device=self.device), true_positives / denominator).mean().item()
    
    def f1score(self):
        precision = self.precision()
        recall = self.recall()
        return 2*precision*recall / (precision + recall)