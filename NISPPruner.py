import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import prune 
from torchmetrics import Accuracy
import os 
import json 

class NISPPruner:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 accuracy: Accuracy, 
                 device: torch.device):
        self.model = model
        self.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.accuracy = accuracy
        self.accuracy.to(device)
        self.importance_scores = self.initialize_importance_scores()

    def initialize_importance_scores(self):
        importance_scores = {}
        for name, param in self.model.named_parameters():
            module = self.get_module_by_name(name)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                importance_scores[name] = torch.zeros_like(param)
        return importance_scores

    def get_module_by_name(self, name):
        modules = name.split('.')
        current_module = self.model
        for module in modules[:-1]:  
            current_module = getattr(current_module, module)
        return current_module

    def compute_importance_scores(self, accumulation_steps: int=64):
        loss_fn = nn.CrossEntropyLoss()
        
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc="Neural Importance Score Computation")

        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets) / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                for name, param in self.model.named_parameters():
                    module = self.get_module_by_name(name)
                    if isinstance(module, (nn.Linear, nn.Conv2d)) and param.grad is not None:
                        self.importance_scores[name] += param.grad.detach().abs() * param.detach().abs()

                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            del loss, outputs, inputs, targets

        pbar.close()
        print(f"Total layers with computed importance scores: {len(self.importance_scores)}")

    def apply_pruning(self, prune_ratio, method='downsampling', downsample_ratio=0.1, chunk_size=1000000):
    
        for name, scores in self.importance_scores.items():
            if method == 'downsampling':
                if downsample_ratio < 1.0:
                    sample_size = int(len(scores.flatten()) * downsample_ratio)
                    sampled_scores = scores.flatten()[torch.randint(len(scores.flatten()), (sample_size,))]
                else:
                    sampled_scores = scores.flatten()

                threshold = torch.quantile(sampled_scores, prune_ratio)
            
            elif method == 'chunk':
                chunk_quantiles = []
                flattened_scores = scores.flatten()
                for i in range(0, len(flattened_scores), chunk_size):
                    chunk = flattened_scores[i:i + chunk_size]
                    chunk_quantiles.append(torch.quantile(chunk, prune_ratio))
                
                threshold = torch.min(torch.tensor(chunk_quantiles))

            else:
                raise ValueError("Invalid method selected. Choose 'downsampling' or 'chunk'.")

            module = self.get_module_by_name(name)
            param_name = name.split('.')[-1]  # Extract the actual parameter name (e.g., 'weight')
            
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Apply the custom pruning method to the module
                prune.custom_from_mask(module, param_name, mask=self._get_mask(module, name, scores, threshold))
            print(f"Pruning applied to layer {name} with threshold: {threshold.item()}")

    def _get_mask(self, module, name, importance_scores, threshold):
        mask = torch.ones_like(importance_scores)
        mask[importance_scores <= threshold] = 0
        return mask
    
    def remove_pruning(self):
        """
        This method removes the pruning reparameterization and makes the pruning permanent.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    # Remove the pruning reparameterization if it exists
                    prune.remove(module, 'weight')
                    print(f"Pruning reparameterization removed from {name}")
                    
                    # If there's a bias term and it was pruned, remove it too
                    if hasattr(module, 'bias') and module.bias is not None:
                        prune.remove(module, 'bias')
                        print(f"Pruning reparameterization removed from bias of {name}")
                except ValueError:
                    # If the module wasn't pruned, skip it
                    print(f"No pruning to remove for {name}")

    def fine_tune(self, epochs=20, learning_rate=1e-4, artifact_path: str="./pruned_artifacts"):

        os.makedirs(artifact_path, exist_ok=True)
        start_epoch = 1
        best_val_acc = 0
        pruning_history = {"train_acc": [], "val_acc": []}

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, epochs + 1):
            train_acc = 0

            pbar = tqdm(self.train_loader, desc=f"Fine tuning: {epoch} / {epochs}")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
               
                optimizer.zero_grad(set_to_none=True)                
                loss.backward()
                optimizer.step()

                self.accuracy.update(outputs, targets)
                acc = self.accuracy.compute().item()
                train_acc += acc / len(self.train_loader)

                pbar.set_postfix(loss=loss.item(), accuracy=acc)
                del inputs, targets, outputs, loss
            
            pruning_history["train_acc"].append(train_acc)
            test_acc = self.evaluate()
            pruning_history["val_acc"].append(test_acc)
            
            with open(os.path.join(artifact_path, "learning_history.json"), "w") as history:
                 json.dump(pruning_history, history, indent=4)
                 
        
        self.remove_pruning()
        torch.save(self.model.state_dict(), os.path.join(artifact_path, "pruned_model.pt"))
        print(f"Fine-tuning completed for {epochs} epochs.")
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.accuracy.reset()
        
        pbar = tqdm(self.val_loader, desc="Test: ", total=len(self.val_loader))
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model(x)
            self.accuracy.update(y_hat, y)
        
        
        acc = self.accuracy.compute().item()
        print(f"----------------------{acc}--------------------------")
        pbar.close()
        return acc

