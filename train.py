from datetime import datetime
import os
import torch
import torch.nn as nn
from transformers import SwinForImageClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss import variance_aware_loss_from_batch

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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: " + str(device))
    model.to(device)

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

        if var:
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
                    else:
                        test_loss += criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_total += labels.size(0)
            test_accuracy = 100 * test_correct / test_total
            print(
                f"Evaluation after Epoch {epoch + 1}: \n"
                f"Eval Loss: {test_loss / len(dataloader_eval):.4f}\n"
                f"Test Accuracy: {test_accuracy:.2f}%")

            writer.add_scalar("Loss/total_eval", test_loss / len(dataloader_eval), epoch)

            if var:
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
