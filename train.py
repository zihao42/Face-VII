from datetime import datetime
import os
import torch
import torch.nn as nn
from transformers import SwinForImageClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss import variance_aware_loss_from_batch, evidential_loss_from_batch
from enn_head import EvidentialClassificationHead

def compute_mu_var(z, labels, eps=1e-6):
    """
    Compute the mean (mu) and diagonal variance (var) for each class based on the 
    current batch features and true labels.
    
    Args:
      z: Tensor of shape (B, M) representing the feature vectors of the current batch (e.g., M=768).
      labels: Tensor of shape (B,) representing the true labels, with values in the range 0 to (K-1).
      eps: A small constant for numerical stability to avoid division by zero.
      
    Returns:
      mu_tensor: Tensor of shape (K', M), where K' is the number of classes present in the batch.
                 Each row is the mean vector for that class.
      var_tensor: Tensor of shape (K', M), where each row is the variance vector (per feature dimension)
                  for that class (with eps added to ensure positivity).
    """
    unique_labels = torch.unique(labels)
    mu_list = []
    var_list = []
    # Process classes in sorted order
    for cl in sorted(unique_labels.tolist()):
        cl_mask = (labels == cl)
        z_cl = z[cl_mask]               # Features belonging to class 'cl', shape (N_cl, M)
        mu_cl = z_cl.mean(dim=0)          # Mean vector, shape (M,)
        var_cl = z_cl.var(dim=0, unbiased=False) + eps  # Variance vector, shape (M,)
        mu_list.append(mu_cl)
        var_list.append(var_cl)
    mu_tensor = torch.stack(mu_list, dim=0)   # Shape (K', M)
    var_tensor = torch.stack(var_list, dim=0)  # Shape (K', M)
    return mu_tensor, var_tensor

def compute_unknown_score(z, labels, eps=1e-6):
    """
    Compute the unknown score S for each sample based on the Mahalanobis distance 
    from the sample to each class center, using the current batch features and labels.
    The function dynamically calculates the class mean and variance, so no external 
    mu or var parameters are needed.
    
    Args:
      z: Tensor of shape (B, M) representing the feature vectors of the current batch (e.g., M=768).
      labels: Tensor of shape (B,) representing the true labels, with values in the range 0 to (K-1).
      eps: A small constant for numerical stability.
      
    Returns:
      S: Tensor of shape (B,) representing the unknown score for each sample.
         A higher score indicates that the sample is further away from all known class centers,
         making it more likely to be considered "unknown".
    """
    # Compute class-wise mean and variance from the batch
    mu, var = compute_mu_var(z, labels, eps)  # mu, var have shape (K', M)
    
    B, M = z.shape
    K_prime, _ = mu.shape
    
    # Expand z to shape (B, 1, M) and mu to shape (1, K', M) for broadcasting
    z_expanded = z.unsqueeze(1)       # Shape: (B, 1, M)
    mu_expanded = mu.unsqueeze(0)       # Shape: (1, K', M)
    
    # Compute the difference between each sample and each class mean
    diff = z_expanded - mu_expanded     # Shape: (B, K', M)
    # Calculate the Mahalanobis distance under the diagonal covariance assumption (ignoring constant factors)
    normalized_squared = diff ** 2 / (var.unsqueeze(0) + eps)  # Shape: (B, K', M)
    # Sum over the feature dimension to get the distance for each sample with each class
    d = normalized_squared.sum(dim=2)   # Shape: (B, K')
    # For each sample, select the minimum distance over all classes as the unknown score S
    S, _ = d.min(dim=1)                 # Shape: (B,)
    
    return S

def train(
        num_epochs,
        eval_gap_epoch,
        num_labels,
        dataloader_train,
        dataloader_eval,
        save_weights_gap_epoch,
        save_weight_dir,
        var=False,
        evi=False):    
    if var:
        print("Variance-based version training starts!")

    if evi:
        print("Evidential version training starts!")
        
    date_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    pooled_representation = {}

    # Hook function to capture the input to the classification layer
    def hook_fn(module, input, output):
        pooled_representation["features"] = output.pooler_output  # Extract input before FC layer

    # Load pre-trained Swin-Tiny model
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    model = SwinForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True,
                                                       num_labels=num_labels)

    model.swin.register_forward_hook(hook_fn)

    # define optimizer
    evi_head = None
    # define EDL head in evi mode
    if evi:
        in_features = model.config.hidden_size
        evi_head = EvidentialClassificationHead(in_features, num_labels)

    if evi:
        # have to take account of the parameters of the evidential head
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(evi_head.parameters()), lr=5e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: " + str(device))
    model.to(device)
    evi_head.to(device)

    # Training loop
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total = 0

        # for variance-based
        total_ldist = 0
        total_lreg = 0
        total_ce_loss = 0
        # I still currently included them in evi for consistency, will revise later

        correct = 0
        for images, labels in tqdm(dataloader_train, desc="Processing"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            features = pooled_representation["features"]
            if var:
                loss, ldist, lreg, ce_loss = variance_aware_loss_from_batch(features, outputs, labels)
                # for variance-based
                total_ldist += ldist.item()
                total_lreg += lreg.item()
                total_ce_loss += ce_loss.item()
            elif evi:
                # for evidential
                evidence = evi_head(features)
                loss, ldist, lreg, ce_loss = evidential_loss_from_batch(evidence, labels)
                # again, just for consistency
                total_ldist += ldist.item()
                total_lreg += lreg.item()
                total_ce_loss += ce_loss.item()
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]:\n "
            f"Loss: {total_loss / len(dataloader_train):.4f}\n"
            f"Accuracy: {100 * correct / total:.2f}%")

        writer.add_scalar("Loss/total", total_loss / len(dataloader_train), epoch)

        if var or evi:
            print(
                f"Loss_ldist: {total_ldist / len(dataloader_train):.4f}, \n"
                f"Loss_lreg: {total_lreg / len(dataloader_train):.4f}, \n"
                f"Loss_ce_loss: {total_ce_loss / len(dataloader_train):.4f}, \n"
            )
            writer.add_scalar("Loss/ldist", total_ldist / len(dataloader_train), epoch)
            writer.add_scalar("Loss/lreg", total_lreg / len(dataloader_train), epoch)
            writer.add_scalar("Loss/ce_loss", total_ce_loss / len(dataloader_train), epoch)

        # Evaluate every 3 epochs
        if (epoch + 1) % eval_gap_epoch == 0:
            model.eval()

            # for edl head if in evi mode
            if evi_head:
                evi_head.eval()
            
            test_correct = 0
            test_loss = 0
            test_total = 0

            # for variance-based
            test_ldist = 0
            test_lreg = 0
            test_ce_loss = 0

            with torch.no_grad():
                for images, labels in dataloader_eval:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).logits
                    features = pooled_representation["features"]
                    if var:
                        losses = variance_aware_loss_from_batch(features, outputs, labels)

                        test_loss += losses[0].item()
                        test_ldist += losses[1].item()
                        test_lreg += losses[2].item()
                        test_ce_loss += losses[3].item()
                    elif evi:
                        evidence = evi_head(features)
                        losses_evi = evidential_loss_from_batch(evidence, labels)
                        test_loss += losses_evi[0].item()
                        # for consistency, will revise later
                        test_ldist += losses_evi[1].item()
                        test_lreg += losses_evi[2].item()
                        test_ce_loss += losses_evi[3].item()
                    else:
                        test_loss += criterion(outputs, labels)

                    # for evi use edl head instead of Swin built-in logits
                    if evi:
                        evidence = evi_head(features)
                        alpha = evidence + 1
                        probs = alpha / alpha.sum(dim=1, keepdim=True)
                        predicted = probs.argmax(dim=1)
                    else:
                        _, predicted = torch.max(outputs, 1)

                    test_correct += (predicted == labels).sum().item()
                    test_total += labels.size(0)

            test_accuracy = 100 * test_correct / test_total
            print(
                f"Evaluation after Epoch {epoch + 1}: \n"
                f"Eval Loss: {test_loss / len(dataloader_eval):.4f}\n"
                f"Test Accuracy: {test_accuracy:.2f}%")

            writer.add_scalar("Loss/total_eval", test_loss / len(dataloader_eval), epoch)

            if var or evi:
                print(
                    f"Loss_ldist: {test_ldist / len(dataloader_eval):.4f}, \n"
                    f"Loss_lreg: {test_lreg / len(dataloader_eval):.4f}, \n"
                    f"Loss_ce_loss: {test_ce_loss / len(dataloader_eval):.4f}, \n"
                )
                writer.add_scalar("Loss/ldist_eval", test_ldist / len(dataloader_eval), epoch)
                writer.add_scalar("Loss/lreg_eval", test_lreg / len(dataloader_eval), epoch)
                writer.add_scalar("Loss/ce_loss_eval", test_ce_loss / len(dataloader_eval), epoch)

        # Save model
        if (epoch + 1) % save_weights_gap_epoch == 0 and epoch + 1 < num_epochs:
            option = ""
            if var:
                option = "_variance"
            elif evi:
                option = "_evidential"

            torch.save(model.state_dict(),
                       os.path.join(save_weight_dir,
                                    "swin_tiny_rafdb_" + date_time_str + "_epoch_" + str(epoch) + option + ".pth"))

    torch.save(model.state_dict(),
               os.path.join(save_weight_dir, "swin_tiny_rafdb_" + date_time_str + "_final" + option + ".pth"))
