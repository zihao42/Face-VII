import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from audio_feature_extract import extract_audio_features
from visual_feature_extract import extract_video_features
from dummy_datasets import DummyTimeSformerDataset, DummyWav2Vec2Dataset
from transformers import TimesformerModel, Wav2Vec2Model
from feature_fusion import MultimodalTransformer


def train_and_evaluate(video_model,
                       audio_model,
                       fusion_model,
                       video_train_loader,
                       video_val_loader,
                       audio_train_loader,
                       audio_val_loader,
                       criterion,
                       optimizer,
                       device,
                       num_epochs=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    video_model.eval()
    audio_model.eval()
    for epoch in range(num_epochs):
        fusion_model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        num_batches = len(video_train_loader)
        video_train_iter = iter(video_train_loader)
        audio_train_iter = iter(audio_train_loader)

        epoch_iter = tqdm(range(num_batches), desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False,
                          unit="batch")
        for _ in epoch_iter:
            waveforms, labels = next(audio_train_iter)
            videos, _ = next(video_train_iter)

            waveforms = waveforms.to(device)
            videos = videos.to(device)  # videos 形状: (batch, T, C, H, W)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                features_v = extract_video_features(videos, video_model)
                features_a = extract_audio_features(waveforms, audio_model, 16)
                assert features_a.shape[-1] == features_v.shape[-1], f"Embedding dims are not matched!" \
                                                                     f"Audio shape: {features_a.shape} v.s. " \
                                                                     f"Video shape: {features_v.shape}"

            logits, fused_feature = fusion_model([features_a, features_v])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            _, preds = torch.max(logits, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)
            epoch_iter.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        fusion_model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        num_batches_val = len(video_val_loader)
        video_val_iter = iter(video_val_loader)
        audio_val_iter = iter(audio_val_loader)

        with torch.no_grad():
            for _ in range(num_batches_val):
                waveforms, labels = next(audio_val_iter)
                videos, _ = next(video_val_iter)
                waveforms = waveforms.to(device)
                videos = videos.to(device)  # videos 形状: (batch, T, C, H, W)
                labels = labels.to(device)

                features_v = extract_video_features(videos, video_model)
                features_a = extract_audio_features(waveforms, audio_model, 16)
                logits, fused_feature = fusion_model([features_a, features_v])
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * waveforms.size(0)
                _, preds = torch.max(logits, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Val   Acc: {epoch_val_acc:.4f}")

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    return fusion_model, metrics


# ===================== 主函数 =====================
def main():
    batch_size = 4
    num_epochs = 3
    learning_rate = 1e-6

    # ---------------------- 设备设置 ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device_str}")

    # 构造数据集
    train_dataset_v = DummyTimeSformerDataset(num_samples=40)
    train_dataset_a = DummyWav2Vec2Dataset(num_samples=40)
    val_dataset_v = DummyTimeSformerDataset(num_samples=10)
    val_dataset_a = DummyWav2Vec2Dataset(num_samples=10)

    # 划分训练集和验证集

    train_loader_v = DataLoader(train_dataset_v, batch_size=batch_size, shuffle=False)
    val_loader_v = DataLoader(val_dataset_v, batch_size=batch_size, shuffle=False)
    train_loader_a = DataLoader(train_dataset_a, batch_size=batch_size, shuffle=False)
    val_loader_a = DataLoader(val_dataset_a, batch_size=batch_size, shuffle=False)

    # 构造模型（每个 CSV 的类别数可能不同）
    print("Loading video model...")
    model_v = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400", output_hidden_states=True
        )

    print("Loading audio model...")
    model_a = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model_v = model_v.to(device)
    model_a = model_a.to(device)
    print("Loading fusion model...")
    model_fusion = MultimodalTransformer(modality_num=2, num_classes=5)
    model_fusion = model_fusion.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_fusion.parameters(), lr=learning_rate)

    # 训练并评估模型
    model, metrics = train_and_evaluate(model_v, model_a, model_fusion,
                                        train_loader_v, val_loader_v,
                                        train_loader_a, val_loader_a,
                                        criterion, optimizer, device,
                                        num_epochs=num_epochs)


if __name__ == '__main__':
    main()
