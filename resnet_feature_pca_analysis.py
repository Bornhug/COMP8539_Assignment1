import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

class ResNetFeatureExtractor(nn.Module):
    """Wrapper for ResNet model to extract features from specific layers"""
    def __init__(self, original_model, feature_layer='avgpool'):
        super().__init__()
        self.original_model = original_model
        self.feature_layer = feature_layer
        self.features = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture features from the specified layer"""
        def get_activation(name):
            def hook(module, input, output):
                if self.feature_layer in name:
                    # For avgpool layer, flatten features
                    if 'avgpool' in name:
                        self.features = output.view(output.size(0), -1)
                    # For other layers, apply global average pooling
                    else:
                        if len(output.shape) == 4:  # [B, C, H, W]
                            self.features = torch.avg_pool2d(output, output.shape[2:]).view(output.size(0), -1)
                        else:
                            self.features = output
            return hook
        
        # Register hook on the specified layer
        for name, module in self.original_model.named_modules():
            if self.feature_layer in name:
                module.register_forward_hook(get_activation(name))
                print(f"Registered hook on layer: {name}")
                break
        
        # If the specified layer is not found, fall back to avgpool
        if not hasattr(self, '_hook_registered'):
            print("Using default avgpool layer for feature extraction")
            if hasattr(self.original_model, 'avgpool'):
                self.original_model.avgpool.register_forward_hook(
                    lambda module, input, output: setattr(self, 'features', 
                        output.view(output.size(0), -1))
                )
    
    def forward(self, x):
        self.features = None
        output = self.original_model(x)
        return output

def create_cifar_resnet(model_type="resnet18", pretrained=True):
    """Create a ResNet model adapted for CIFAR-10"""
    if model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Modify the first conv layer for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool layer since CIFAR-10 images are small
    model.maxpool = nn.Identity()
    
    # Modify final fully connected layer for 10 CIFAR-10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def extract_resnet_features(model, data_loader, device, max_samples=2000):
    """Extract features using the ResNet model"""
    model.eval()
    all_features = []
    all_labels = []
    
    print(f"Starting feature extraction, max samples: {max_samples}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * data.size(0) >= max_samples:
                break
                
            data, target = data.to(device), target.to(device)
            _ = model(data)  # Forward pass to trigger hooks
            
            if model.features is not None:
                features = model.features.cpu().numpy()
                all_features.append(features)
                all_labels.append(target.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")
    
    if not all_features:
        raise ValueError("No features extracted. Please check if hooks are working properly.")
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return all_features, all_labels

def perform_pca_analysis(features, n_components=50):
    """Perform PCA analysis on the extracted features"""
    print("\n" + "="*60)
    print("Starting PCA Analysis")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform PCA
    n_components = min(n_components, features.shape[1])
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"Explained variance ratio for first 5 components:")
    for i in range(min(5, len(explained_variance_ratio))):
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]:.2%})")
    
    print(f"\nCumulative explained variance ratio:")
    for i in range(min(10, len(cumulative_variance_ratio))):
        print(f"  First {i+1} components: {cumulative_variance_ratio[i]:.4f} ({cumulative_variance_ratio[i]:.2%})")
    
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
    
    print(f"\nNumber of components for 95% variance: {n_components_95}")
    print(f"Number of components for 99% variance: {n_components_99}")
    
    return pca, features_pca, features_scaled, scaler

def plot_pca_results(pca, features_pca, labels, outdir, model_type, feature_layer):
    """Plot PCA results - Only keep 4 most important charts"""
    os.makedirs(outdir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Explained variance ratio - Most important chart
    plt.subplot(2, 2, 1)
    n_components_plot = min(20, len(pca.explained_variance_ratio_))
    plt.bar(range(1, n_components_plot + 1), 
             pca.explained_variance_ratio_[:n_components_plot])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'{model_type} - Explained Variance Ratio\n(Layer: {feature_layer})')
    plt.grid(True, alpha=0.3)
    
    # 2. Cumulative variance ratio - Show how many components needed
    plt.subplot(2, 2, 2)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), 
             cumulative_variance, 'bo-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance', linewidth=2)
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.title(f'{model_type} - Cumulative Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Scatter plot of first 2 PCs - Show data distribution
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=30)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title(f'{model_type} - Scatter Plot of First 2 PCs')
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    
    # 4. 3D Scatter plot of first 3 PCs - Show 3D distribution
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    scatter3d = ax3d.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], 
                             c=labels, alpha=0.7, cmap='tab10', s=20)
    ax3d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax3d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax3d.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax3d.set_title(f'{model_type} - 3D Scatter Plot of First 3 PCs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{model_type}_{feature_layer}_pca_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA analysis plots saved to: {outdir}")
    print("Only kept 4 most important charts: explained variance ratio, cumulative variance ratio, first 2 PCs scatter plot, first 3 PCs 3D scatter plot")

def save_pca_results(pca, features_pca, labels, outdir, model_type, feature_layer):
    """Save PCA results to disk"""
    os.makedirs(outdir, exist_ok=True)
    
    np.save(os.path.join(outdir, f'{model_type}_{feature_layer}_features_pca.npy'), features_pca)
    np.save(os.path.join(outdir, f'{model_type}_{feature_layer}_labels.npy'), labels)
    np.save(os.path.join(outdir, f'{model_type}_{feature_layer}_pca_components.npy'), pca.components_)
    np.save(os.path.join(outdir, f'{model_type}_{feature_layer}_explained_variance_ratio.npy'), 
            pca.explained_variance_ratio_)
    
    n_components = min(50, features_pca.shape[1])
    pca_df = pd.DataFrame(features_pca[:, :n_components], 
                          columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['class'] = labels
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    pca_df['class_name'] = np.array(class_names)[labels]
    pca_df.to_csv(os.path.join(outdir, f'{model_type}_{feature_layer}_pca_results.csv'), index=False)
    
    stats_df = pd.DataFrame({
        'PC': range(1, len(pca.explained_variance_ratio_) + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    stats_df.to_csv(os.path.join(outdir, f'{model_type}_{feature_layer}_pca_stats.csv'), index=False)
    
    print(f"PCA results saved to: {outdir}")

def main():
    """Main function"""
    model_type = "resnet18"  
    feature_layer = "avgpool"  
    batch_size = 128
    max_samples = 2000  
    outdir = f"runs/{model_type}_{feature_layer}_feature_pca"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Creating {model_type} model...")
    model = create_cifar_resnet(model_type, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created successfully:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    print("Loading CIFAR-10 test dataset...")
    test_dataset = datasets.CIFAR10('./data', train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model_with_hooks = ResNetFeatureExtractor(model, feature_layer)
    model_with_hooks = model_with_hooks.to(device)
    
    print("Extracting features...")
    features, labels = extract_resnet_features(model_with_hooks, test_loader, device, max_samples)
    
    print("Performing PCA analysis...")
    pca, features_pca, features_scaled, scaler = perform_pca_analysis(features, n_components=50)
    
    print("Plotting PCA results...")
    plot_pca_results(pca, features_pca, labels, outdir, model_type, feature_layer)
    
    print("Saving PCA results...")
    save_pca_results(pca, features_pca, labels, outdir, model_type, feature_layer)
    
    print("\n" + "="*60)
    print("PCA analysis complete!")
    print("="*60)
    print(f"Total variance explained by first 10 PCs: {np.sum(pca.explained_variance_ratio_[:10]):.3f}")
    print(f"Results saved to: {outdir}")

if __name__ == "__main__":
    main()
