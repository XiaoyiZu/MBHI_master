import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Configurations
class Config:
    data_path = "./data/ABA_SHIFT_CLASSIFY"
    batch_size = 16
    num_points = 8000
    num_classes = 3
    num_epochs = 600
    learning_rate = 0.001
    save_path = "./models/best_model.pth"

# Preprocess data
class PointCloudDataset(Dataset):
    def __init__(self, files, labels, num_points):
        self.files = files
        self.labels = torch.LongTensor(labels)
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.loadtxt(self.files[idx], delimiter=',')
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=0)
            points = data[:, :6]  # x,y,z,r,g,b

            if len(points) == 0:
                print(f"Warning: Empty point cloud in {self.files[idx]}")
                points = np.zeros((self.num_points, 6))

            if len(points) > self.num_points:
                idxs = np.linspace(0, len(points) - 1, self.num_points, dtype=np.int32)
                points = points[idxs]
            elif len(points) < self.num_points:
                repeat = self.num_points // len(points) + 1
                points = np.tile(points, (repeat, 1))
                points = points[:self.num_points]

            points[:, :3] = points[:, :3] - np.mean(points[:, :3], axis=0)
            max_norm = np.max(np.linalg.norm(points[:, :3], axis=1)) + 1e-6
            points[:, :3] = points[:, :3] / max_norm

            points = np.nan_to_num(points)
            return torch.FloatTensor(points), self.labels[idx]

        except Exception as e:
            print(f"Error processing file {self.files[idx]}: {str(e)}")
            return torch.zeros(self.num_points, 6), 0


# GAM module
class GeometryAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.linear1 = nn.Linear(channels, channels * 2)
        self.linear2 = nn.Linear(channels * 2, channels * 2)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        B, C, N = x.shape

        assert C == self.channels, f"Input channels {C} != expected {self.channels}"

        # Calculate global feature
        global_feat = torch.mean(x, dim=2)  # (B, C)

        # Calculate transformation parameters
        params = F.relu(self.linear1(global_feat))  # (B, 2*C)
        params = self.linear2(params)  # (B, 2*C)

        gamma = params[:, :self.channels].reshape(B, C, 1)  # (B, C, 1)
        beta = params[:, self.channels:].reshape(B, C, 1)  # (B, C, 1)

        # Apply transformation
        x = gamma * x + beta
        return self.bn(x)


# PointMLP module
class PointMLPBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
        )
        self.gam = GeometryAffine(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.mlp(x)
        x = self.gam(x)
        return F.relu(x + shortcut)

# Multi-scale feature extraction
class MultiScaleFeature(nn.Module):
    def __init__(self):
        super().__init__()
        # Global feature
        self.global_mlp1 = PointMLPBlock(6, 64)
        self.global_mlp2 = PointMLPBlock(64, 128)
        self.global_mlp3 = PointMLPBlock(128, 256)

        # Meso feature
        self.meso_mlp1 = PointMLPBlock(6, 64)
        self.meso_mlp2 = PointMLPBlock(64, 128)

        # Detail feature
        self.detail_mlp = PointMLPBlock(6, 64)

    def forward(self, x):
        B, C, N = x.shape
        print(f"Initial input shape: {x.shape}")

        # Global feature
        global_feat = self.global_mlp1(x)
        global_feat = self.global_mlp2(global_feat)
        global_feat = self.global_mlp3(global_feat)
        global_feat = torch.max(global_feat, dim=2)[0]  # (B, 256)

        # Meso feature
        meso_feat = self.meso_mlp1(x)
        meso_feat = self.meso_mlp2(meso_feat)
        meso_feat = torch.max(meso_feat, dim=2)[0]  # (B, 128)

        # Detail feature
        detail_feat = self.detail_mlp(x)
        detail_feat = torch.max(detail_feat, dim=2)[0]  # (B, 64)

        return global_feat, meso_feat, detail_feat


# General architecture
class PointMLPClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = MultiScaleFeature()
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)

        global_feat, meso_feat, detail_feat = self.feature_extractor(x)
        features = torch.cat([global_feat, meso_feat, detail_feat], dim=1)

        return self.classifier(features), (global_feat, meso_feat, detail_feat)


# Data preparation
def prepare_data():
    class_folders = ['Heishui_group', 'Jiuzhaigou_group', 'Songpan_group']
    all_files = []
    all_labels = []

    for label, folder in enumerate(class_folders):
        folder_path = os.path.join(Config.data_path, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
        files = [f for f in files if os.path.getsize(f) > 0]
        all_files.extend(files)
        all_labels.extend([label] * len(files))

    # Check data
    if len(all_files) == 0:
        raise ValueError("No valid data files found in the specified path!")

    _, test_files, _, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    train_files = all_files.copy()
    train_labels = all_labels.copy()

    return train_files, test_files, train_labels, test_labels


# Train model
def train():
    try:
        train_files, test_files, train_labels, test_labels = prepare_data()

        train_dataset = PointCloudDataset(train_files, train_labels, Config.num_points)
        test_dataset = PointCloudDataset(test_files, test_labels, Config.num_points)

        # Check data loading
        sample_points, sample_label = train_dataset[0]
        print(f"Sample data shape: {sample_points.shape}, Label: {sample_label}")

        train_loader = DataLoader(train_dataset,
                                  batch_size=Config.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)

        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)

        model = PointMLPClassifier(Config.num_classes)
        print(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

        best_test_loss = float('inf')

        for epoch in range(Config.num_epochs):
            # Training stage
            model.train()
            train_loss, train_correct = 0, 0

            for points, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
                points, labels = points.to(device), labels.to(device)
                optimizer.zero_grad()

                if torch.isnan(points).any():
                    print("Warning: NaN in input points!")

                logits, _ = model(points)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (logits.argmax(dim=1) == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_dataset)

            # Test stage
            model.eval()
            test_loss, test_correct = 0, 0

            with torch.no_grad():
                for points, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1} Test"):
                    points, labels = points.to(device), labels.to(device)
                    logits, _ = model(points)
                    loss = criterion(logits, labels)

                    test_loss += loss.item()
                    test_correct += (logits.argmax(dim=1) == labels).sum().item()

            test_loss /= len(test_loader)
            test_acc = test_correct / len(test_dataset)

            print(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")

            # Save optimal model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), Config.save_path)
                print(f"Model saved to {Config.save_path}")
        torch.save(model.state_dict(), Config.save_path.replace('best','last'))
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise


if __name__ == "__main__":
    train()




