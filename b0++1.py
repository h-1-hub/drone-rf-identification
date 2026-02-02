import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, random_split, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from torchvision.models import efficientnet_b0
import matplotlib
from collections import Counter
from torch.utils.data import WeightedRandomSampler

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
IMAGE_SAVE_DIR = "9"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
print(f"�� 图片将保存到: {IMAGE_SAVE_DIR}")


# ====================== Dataset ======================
class DroneDataset(Dataset):
    def __init__(self, root_path, is_test=False, known_only=True, unknown_label=18):
        self.root = Path(root_path)
        self.is_test = is_test
        self.known_only = known_only
        self.samples = []
        self.unknown_label = unknown_label
        self._load_samples()
        self._print_stats()

    def _load_samples(self):
        if "unknown" in str(self.root).lower():
            for class_dir in sorted(self.root.iterdir()):
                if class_dir.is_dir():
                    files = list(class_dir.glob("*.npy"))
                    self.samples.extend([(f, self.unknown_label) for f in files])
        else:
            for class_dir in sorted(self.root.iterdir()):
                if class_dir.is_dir():
                    try:
                        class_id = int(class_dir.name)
                        if self.known_only:
                            if class_id > 17:
                                continue
                            mapped_label = class_id
                        else:
                            mapped_label = class_id if class_id <= 17 else self.unknown_label
                    except ValueError:
                        mapped_label = self.unknown_label
                    files = list(class_dir.glob("*.npy"))
                    self.samples.extend([(f, mapped_label) for f in files])

    def _print_stats(self):
        from collections import defaultdict
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1
        print(f"\n{'测试集' if self.is_test else '训练集'}统计:")
        print(f"总样本数: {len(self.samples)}")
        print("类别分布:")
        for label, count in sorted(counts.items()):
            print(f"  类别 {label}: {count} 样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path).astype(np.float32)
        mean = data.mean()
        std = data.std() + 1e-8
        data = (data - mean) / std
        data = torch.from_numpy(data).unsqueeze(0)  # [1, H, W]

        if not self.is_test:
            # 更温和的数据增强
            if np.random.rand() > 0.5:
                # 轻微的时间偏移（更小的范围）
                shift = np.random.randint(-3, 4)
                if shift != 0:
                    data = torch.roll(data, shifts=shift, dims=2)

            if np.random.rand() > 0.7:
                # 更小的频谱掩码
                mask_len = np.random.randint(3, 8)
                mask_start = np.random.randint(0, max(1, data.shape[2] - mask_len))
                data[:, :, mask_start:mask_start + mask_len] = data.mean()

            if np.random.rand() > 0.5:
                # 添加轻微的高斯噪声
                noise_std = np.random.uniform(0.01, 0.05)
                data += torch.randn_like(data) * noise_std

            if np.random.rand() > 0.7:
                # 频谱缩放
                scale = np.random.uniform(0.8, 1.2)
                data = data * scale

        return data, label


# ====================== SE Block ======================
class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ====================== OSAM Loss ======================
class OSAMLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_attr=0.1, lambda_repl=0.2, margin=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_attr = lambda_attr
        self.lambda_repl = lambda_repl
        self.margin = margin

        self.register_buffer('centers', torch.zeros(num_classes, feature_dim))
        self.register_buffer('radii', torch.ones(num_classes) * 1.0)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, logits, labels):
        ce = self.ce_loss(logits, labels)

        if self.training:
            with torch.no_grad():
                for lbl in labels.unique():
                    mask = (labels == lbl)
                    if mask.sum() == 0: continue
                    feat_cls = features[mask]
                    new_center = feat_cls.mean(dim=0)
                    self.centers[lbl] = 0.9 * self.centers[lbl] + 0.1 * new_center.detach()
                    dists = torch.norm(feat_cls - self.centers[lbl], dim=1)
                    mean_dist = dists.mean()
                    self.radii[lbl] = 0.9 * self.radii[lbl] + 0.1 * mean_dist

        centers_exp = self.centers[labels]
        dist = torch.norm(features - centers_exp, dim=1)
        L_attr = (dist ** 2).mean()

        R = self.radii[labels] + self.margin
        excess = torch.clamp(dist - R, min=0.0)
        L_repl = (excess ** 2).mean()

        total_loss = ce + self.lambda_attr * L_attr + self.lambda_repl * L_repl
        return total_loss, {"ce": ce.item(), "L_attr": L_attr.item(), "L_repl": L_repl.item(),
                            "avg_radius": self.radii.mean().item()}


# ===========================================
# 分支定义
# ===========================================
class SpectralConvBranch(nn.Module):
    """轻量版分支：Conv1D提取时序位置特征"""

    def __init__(self, in_ch=1, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = x.mean(dim=2)  # 频率平均 -> 保留时间序列
        x = self.net(x).flatten(1)
        return self.fc(x)


class SpectralTransformerBranch(nn.Module):
    """强版分支：Transformer捕获全局频域关系"""

    def __init__(self, in_ch=1, feature_dim=128, num_heads=8, depth=2):
        super().__init__()
        self.conv_embed = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(64, feature_dim)
        # 修正：添加 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True  # 添加这个参数
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.conv_embed(x)  # [B, 64, H/2, W/2]
        B, C, H, W = x.shape
        x = x.mean(dim=2).permute(0, 2, 1)  # [B, W/2, 64] - 改为 batch_first
        x = self.proj(x)
        x = self.encoder(x)
        return x.mean(dim=1)  # [B, feature_dim] - 改为 dim=1

class FusionHead(nn.Module):
    """融合两个分支特征"""

    def __init__(self, in_dim1, in_dim2, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim1 + in_dim2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, f1, f2):
        return self.fc(torch.cat([f1, f2], dim=1))


# ===========================================
# 主体网络：双分支 EfficientNet + Conv/Transformer
# ===========================================
class MultiDomainEfficientNet(nn.Module):
    def __init__(self, feature_dim=128, num_classes=18, branch_type='transformer'):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(1, first_conv.out_channels,
                                            kernel_size=first_conv.kernel_size,
                                            stride=first_conv.stride,
                                            padding=first_conv.padding,
                                            bias=False)
        self.texture_branch = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1))
        self.position_branch = SpectralTransformerBranch(in_ch=1, feature_dim=feature_dim) \
            if branch_type == 'transformer' else SpectralConvBranch(in_ch=1, feature_dim=feature_dim)
        self.fusion = FusionHead(1280, feature_dim, out_dim=feature_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat_tex = self.texture_branch(x).flatten(1)
        feat_pos = self.position_branch(x)
        fused = self.fusion(feat_tex, feat_pos)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return fused, logits


# ===========================================
# 训练器 + 开集评估
# ===========================================
class Trainer:
    def __init__(self, model, num_classes=18, feature_dim=128, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 改进 OSAMLoss 参数
        self.criterion = OSAMLoss(
            num_classes,
            feature_dim,
            lambda_attr=0.02,  # 降低权重
            lambda_repl=0.4,  # 降低权重
            margin=0.086 # 减小边界
        ).to(device)

        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 降低学习率

        # ✅ 修正：学习率调度器 - ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )  # 移除 verbose 参数

        # 早停机制相关
        self.best_val_loss = float('inf')
        self.patience = 15
        self.counter = 0
        self.best_model_state = None

        self.class_means = {}
        self.class_inv_covs = {}
        self.thresholds = {}
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []

    # ============================================================
    def _validate(self, loader):
        self.model.eval()
        total, correct, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                f, logit = self.model(x)
                loss, _ = self.criterion(f, logit, y)
                total_loss += loss.item() * x.size(0)
                pred = logit.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        return total_loss / total, correct / total

    # ============================================================
    def train(self, train_loader, val_loader, epochs=40):
        print("开始训练...")
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                f, logit = self.model(x)
                loss, _ = self.criterion(f, logit, y)
                loss.backward()

                # === 添加梯度裁剪 ===
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)

                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                pred = logit.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)

            # 验证
            val_loss, val_acc = self._validate(val_loader)
            train_loss, train_acc = total_loss / total, correct / total

            # === 学习率调度器调用 ===
            self.scheduler.step(val_loss)  # 这里调用，基于验证损失

            # === 早停机制 ===
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
                print(f"✅ 最佳模型更新: val_loss = {val_loss:.4f}")
            else:
                self.counter += 1
                print(f"⚠️ 早停计数: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print("�� 早停触发，恢复最佳模型")
                self.model.load_state_dict(self.best_model_state)
                break

            # 记录历史
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            lr_now = self.optimizer.param_groups[0]['lr']  # 获取当前学习率
            print(f"Epoch {epoch + 1}: "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val {val_loss:.4f}/{val_acc:.4f} | LR={lr_now:.6f}")

        self._compute_class_statistics(train_loader)

    # ============================================================
    def _compute_class_statistics(self, loader):
        """在 GPU 上提取特征，在 CPU 上计算 Mahalanobis 统计量，并绘制类半径图"""
        print("正在计算类中心与协方差矩阵...")
        self.model.eval()
        all_feats, all_labels = [], []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                f, _ = self.model(x)
                all_feats.append(f.cpu())
                all_labels.append(y.cpu())

        feats = torch.cat(all_feats).numpy()
        labels = torch.cat(all_labels).numpy()
        centers = self.criterion.centers.cpu().numpy()

        self.class_means = {}
        self.class_inv_covs = {}
        self.thresholds = {}

        for c in range(self.num_classes):
            mask = (labels == c)
            if not np.any(mask):
                continue
            cls_feats = feats[mask]
            mu = centers[c]
            cov = np.cov(cls_feats, rowvar=False)
            if cov.ndim == 0:
                cov = np.eye(self.feature_dim) * 1e-4
            else:
                cov += np.eye(cov.shape[0]) * 1e-5
            try:
                inv_cov = np.linalg.pinv(cov)
            except:
                inv_cov = np.eye(self.feature_dim)

            diff = cls_feats - mu
            maha_dists = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))

            # 阈值调整：mean + 1.5 * std
            theta = np.mean(maha_dists) + 3.3 * np.std(maha_dists)

            self.class_means[c] = mu
            self.class_inv_covs[c] = inv_cov
            self.thresholds[c] = theta

        print(f"✅ 共 {len(self.class_means)} 个类的统计信息已计算完成")
        for c in range(self.num_classes):
            if c in self.thresholds:
                print(f"  类 {c}: 半径={self.criterion.radii[c]:.3f} | 阈值={self.thresholds[c]:.3f}")

        # ✅ 新增：绘制每类半径（radii）分布条形图
        try:
            radii_vals = self.criterion.radii.cpu().numpy()
            classes = np.arange(self.num_classes)

            plt.figure(figsize=(10, 5))
            sns.barplot(x=classes, y=radii_vals, palette="coolwarm")
            plt.title("每类特征半径（Radii）分布", fontsize=14)
            plt.xlabel("类别索引", fontsize=12)
            plt.ylabel("半径值 (r)", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 保存图片而不是显示
            plt.savefig(os.path.join(IMAGE_SAVE_DIR, "class_radii_distribution.png"), dpi=150, bbox_inches='tight')
            plt.close()  # 关闭图形释放内存

            print(f"✅ 类半径分布图已保存到: {os.path.join(IMAGE_SAVE_DIR, 'class_radii_distribution.png')}")

            # ✅ 可选：打印最宽的类（调试信息）
            widest_idx = int(np.argmax(radii_vals))
            print(f"⚠️ 半径最大类：Class {widest_idx}，r = {radii_vals[widest_idx]:.3f}")
            print("提示：半径越大，说明该类分布更松散，容易与未知类混淆。")

        except Exception as e:
            print(f"⚠️ 绘制 radii 条形图时出错: {e}")

    # ============================================================
    def evaluate(self, test_loader):
        print("进行开集评估...")
        self.model.eval()
        all_feats, all_labels, all_preds = [], [], []

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Evaluating"):
                x = x.to(self.device)
                f, logits = self.model(x)
                f_np = f.cpu().numpy()
                logit_np = logits.cpu().numpy()
                pred_cls = logit_np.argmax(axis=1)

                maha_dists = []
                for c in range(self.num_classes):
                    mu = self.class_means.get(c, np.zeros(self.feature_dim))
                    inv_cov = self.class_inv_covs.get(c, np.eye(self.feature_dim))
                    diff = f_np - mu
                    dist = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))
                    maha_dists.append(dist)
                maha_dists = np.stack(maha_dists, axis=1)
                min_dists = maha_dists.min(axis=1)
                nearest_class = maha_dists.argmin(axis=1)

                final_preds = []
                for i in range(len(pred_cls)):
                    if min_dists[i] > self.thresholds.get(nearest_class[i], 10.0):
                        final_preds.append(self.num_classes)
                    else:
                        final_preds.append(nearest_class[i])

                all_feats.append(f_np)
                all_labels.append(y.numpy())
                all_preds.extend(final_preds)

        features = np.vstack(all_feats)
        labels_true = np.hstack(all_labels)
        labels_pred = np.array(all_preds)
        labels_true[labels_true >= self.num_classes] = self.num_classes

        self._plot_tsne(features, labels_true, labels_pred, n_samples=1000)

        target_names = [f"Class {i}" for i in range(self.num_classes)] + ["Unknown"]
        report = classification_report(labels_true, labels_pred, target_names=target_names, digits=4)
        print("\n分类报告:")
        print(report)

        cm = confusion_matrix(labels_true, labels_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.title("Confusion Matrix (Open-set)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存混淆矩阵
        plt.savefig(os.path.join(IMAGE_SAVE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 混淆矩阵已保存到: {os.path.join(IMAGE_SAVE_DIR, 'confusion_matrix.png')}")

        return labels_true, labels_pred, features

    def _plot_tsne(self, features, true_labels, pred_labels, n_samples=1000):
        """绘制 t-SNE 图，左边真实标签，右边预测标签"""
        # 下采样避免过载
        indices = np.random.choice(len(features), size=min(n_samples, len(features)), replace=False)
        feats_sample = features[indices]
        true_sample = true_labels[indices]
        pred_sample = pred_labels[indices]

        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        feats_2d = tsne.fit_transform(feats_sample)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 所有类别的颜色映射（前18类 + Unknown类）
        from matplotlib.colors import ListedColormap
        cmap_known = plt.cm.tab20
        colors_known = [cmap_known(i % 20) for i in range(self.num_classes)]  # 18个已知类
        color_unknown = "red"
        colors_all = colors_known + [color_unknown]  # 总共19个颜色

        # 修正：创建离散的颜色映射
        cmap_custom = ListedColormap(colors_all)

        # 标签范围 - 修正：应该是0-17 + Unknown(18)，总共19个
        label_set = list(range(self.num_classes)) + [self.num_classes]  # [0,1,...,17,18]
        label_names = [f"Class {i}" for i in range(self.num_classes)] + ["Unknown"]

        # 左图：真实标签
        sc1 = ax1.scatter(feats_2d[:, 0], feats_2d[:, 1], c=true_sample,
                          cmap=cmap_custom, vmin=0, vmax=self.num_classes, s=50, alpha=0.8)
        ax1.set_title("t-SNE: Ground Truth Labels")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

        # 修正颜色条设置
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_ticks(label_set)
        cbar1.set_ticklabels(label_names)

        # 右图：预测标签
        sc2 = ax2.scatter(feats_2d[:, 0], feats_2d[:, 1], c=pred_sample,
                          cmap=cmap_custom, vmin=0, vmax=self.num_classes, s=50, alpha=0.8)
        ax2.set_title("t-SNE: Predicted Labels")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")

        # 修正颜色条设置
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_ticks(label_set)
        cbar2.set_ticklabels(label_names)

        # 设置未知类标签颜色为红色
        try:
            # 对于垂直颜色条
            yticklabels1 = cbar1.ax.get_yticklabels()
            yticklabels2 = cbar2.ax.get_yticklabels()
            if len(yticklabels1) > 0:
                yticklabels1[-1].set_color("red")
            if len(yticklabels2) > 0:
                yticklabels2[-1].set_color("red")
        except (IndexError, AttributeError):
            pass

        plt.suptitle("t-SNE Visualization of Feature Embeddings", fontsize=16)
        plt.tight_layout()

        # 保存t-SNE图
        tsne_path = os.path.join(IMAGE_SAVE_DIR, "tsne_comparison.png")
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ t-SNE 图已保存为 {tsne_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_root = Path(r"/media/dell/My PSSD/DroneRFb-Spectra/experiment_groups/9")

    full_train_dataset = DroneDataset(os.path.join(data_root, "9-known_for_train"), is_test=False, known_only=True)
    val_size = int(len(full_train_dataset) * 0.2)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_known_set = DroneDataset(data_root / "9-known_for_test", is_test=True, known_only=True)
    test_unknown_set = DroneDataset(data_root / "9-unknown", is_test=True, known_only=False)
    test_set = ConcatDataset([test_known_set, test_unknown_set])

    batch_size = 8
    # === 平衡采样：针对 class 0 样本稀少问题 ===
    print("计算每类样本数量以进行加权采样...")
    labels_all = [y for _, y in train_dataset]  # 取出训练集标签
    class_counts = Counter(labels_all)
    total_samples = len(train_dataset)

    # 计算每个样本的权重：稀有类 → 高权重
    weights = [1.0 / class_counts[y] for y in labels_all]
    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)

    # 使用 sampler 替代 shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    print("✅ 已启用 WeightedRandomSampler（类别平衡采样）")

    # 验证集和测试集不变
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = MultiDomainEfficientNet(feature_dim=128, num_classes=18, branch_type='transformer')
    trainer = Trainer(model, num_classes=18, feature_dim=128, device=device)

    # === 训练 ===
    trainer.train(train_loader, val_loader, epochs=40)

    # === 保存训练曲线 ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_loss_history, label="Train Loss")
    plt.plot(trainer.val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_acc_history, label="Train Acc")
    plt.plot(trainer.val_acc_history, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy ")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 保存训练曲线
    train_curve_path = os.path.join(IMAGE_SAVE_DIR, "train_curve.png")
    plt.savefig(train_curve_path)
    plt.close()
    print(f"训练曲线已保存为 {train_curve_path}")

    # === 开集评估 ===
    trainer.evaluate(test_loader)

    # 保存模型
    model_save_path = os.path.join(IMAGE_SAVE_DIR, "drone_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ 模型已保存为 {model_save_path}")

    print(f"\n�� 所有结果已保存到 '{IMAGE_SAVE_DIR}' 文件夹！")