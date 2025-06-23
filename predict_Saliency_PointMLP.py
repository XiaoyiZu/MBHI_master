import os
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from matplotlib.gridspec import GridSpec
from train_PointMLP import PointMLPClassifier, Config


class SaliencyAnalyzer:
    def __init__(self):
        self.config = Config()
        self.config.class_names = ['Heishui_group', 'Jiuzhaigou_group', 'Songpan_group']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.model.eval()
        self.weights = {'global': 0.2, 'meso': 0.2, 'detail': 0.6}

    def load_model(self):
        """Load pretrained model"""
        self.model = PointMLPClassifier(self.config.num_classes)
        self.model.load_state_dict(
            torch.load('./models/best_model.pth',
                       map_location=self.device)
        )
        self.model = self.model.to(self.device)

    def _process_raw_data(self, input_path):
        raw_data = np.loadtxt(input_path, delimiter=',')
        if raw_data.shape[1] < 6:
            raw_data = np.hstack([raw_data, np.zeros((len(raw_data), 6 - raw_data.shape[1]))])
        return raw_data

    def _record_normalization_params(self, points):
        """Trace normalization parameters"""
        self.original_mean = points[:, :3].mean(axis=0)
        self.original_scale = np.linalg.norm(points[:, :3], axis=1).max() + 1e-6
        return (points[:, :3] - self.original_mean) / self.original_scale

    def _reverse_normalization(self, points):
        return points * self.original_scale + self.original_mean

    def _normalize_saliency(self, saliency):
        """Normalize saliency"""
        saliency = np.log1p(saliency)
        lower = np.percentile(saliency, 5)
        upper = np.percentile(saliency, 95)
        saliency = np.clip(saliency, lower, upper)
        return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)

    def compute_saliency(self, input_path):
        """Multi-scale weight aggregation"""
        raw_data = self._process_raw_data(input_path)
        original_xyz = raw_data[:, :3].copy()

        n_points = len(raw_data)
        if n_points > self.config.num_points:
            sample_indices = np.linspace(0, n_points - 1, self.config.num_points, dtype=np.int32)
            sampled_data = raw_data[sample_indices]
            is_subsampled = True
        else:
            sampled_data = raw_data
            is_subsampled = False

        # Nornalize and record parameters
        normalized_xyz = self._record_normalization_params(sampled_data)
        input_points = sampled_data.copy()
        input_points[:, :3] = normalized_xyz

        # Saliency calculation
        input_tensor = torch.FloatTensor(input_points).unsqueeze(0).to(self.device).requires_grad_()
        with torch.enable_grad():
            logits, features = self.model(input_tensor)
            saliency_maps = {}
            multi_scale_saliencies = {}

            for name, feat in zip(['global', 'meso', 'detail'], features):
                self.model.zero_grad()
                feat[0].sum().backward(retain_graph=True)
                raw_saliency = input_tensor.grad.abs().sum(2).squeeze().cpu().numpy()
                norm_saliency = self._normalize_saliency(raw_saliency)
                saliency_maps[name] = norm_saliency
                input_tensor.grad = None

        # Spatial alignment
        final_results = {}
        restored_xyz = self._reverse_normalization(normalized_xyz) if is_subsampled else original_xyz

        for scale in ['global', 'meso', 'detail']:
            if is_subsampled:
                interpolated = self._interpolate_saliency(original_xyz, restored_xyz, saliency_maps[scale])
            else:
                interpolated = saliency_maps[scale]
            final_results[scale] = interpolated

        # Compute weighted combined saliency
        combined_saliency = sum(
            self.weights[k] * final_results[k]
            for k in self.weights.keys()
        )

        return {
            'original_xyz': original_xyz,
            'saliency_combined': combined_saliency,
            'saliency_global': final_results['global'],
            'saliency_meso': final_results['meso'],
            'saliency_detail': final_results['detail'],
            'class_name': self.config.class_names[logits.argmax().item()],
            'logits': np.round(logits.detach().cpu().numpy().squeeze(), 3)
        }

    def _interpolate_saliency(self, target_pts, source_pts, source_values, k=10):
        tree = cKDTree(source_pts)
        distances, indices = tree.query(target_pts, k=k)

        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1)[:, np.newaxis]  # 归一化权重

        # Weighted average saliency
        interpolated = np.sum(source_values[indices] * weights, axis=1)
        return interpolated

    def visualize_saliency_3d(self, xyz, saliency, title, save_path):
        """Visualize saliency"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if len(xyz) > 500000:
            idx = np.random.choice(len(xyz), 500000, replace=False)
            xyz = xyz[idx]
            saliency = saliency[idx]

        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        c=saliency, cmap='RdYlBu_r',
                        s=6,
                        alpha=1, linewidths=0)

        ax.set_title(title, pad=10, fontsize=10)
        plt.colorbar(sc, shrink=0.5, label='Saliency Score')
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def batch_process(input_dir, output_dir):
    analyzer = SaliencyAnalyzer()
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm([f for f in os.listdir(input_dir) if f.endswith('.txt')]):

        input_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]

        # Compute saliency
        result = analyzer.compute_saliency(input_path)

        # Save complete data (including multi-scale saliency)
        output_data = np.column_stack([
            result['original_xyz'],
            result['saliency_global'],
            result['saliency_meso'],
            result['saliency_detail'],
            result['saliency_combined']
        ])

        np.savetxt(
            os.path.join(output_dir, f"{base_name}_saliency.txt"),
            output_data,
            fmt="%.6f"
        )

        # Generate saliency results
        analyzer.visualize_saliency_3d(
            result['original_xyz'],
            result['saliency_combined'],
            f"{base_name}\nPred: {result['class_name']}",
            os.path.join(output_dir, f"{base_name}_3d.png")
        )


if __name__ == "__main__":
    input_dir = "./data/all_to_predict"
    output_dir = "./data/saliency_results226"

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    batch_process(input_dir, output_dir)
