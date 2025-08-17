import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

class ViTCLSTokenExtractor(nn.Module):
    """ViT model wrapper for extracting CLS token embeddings"""
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.cls_token_embeddings = None
        self.patch_embeddings = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture CLS token and patch embeddings"""
        def get_cls_token_hook(name):
            def hook(module, input, output):
                # For transformer blocks, capture the output after self-attention
                if 'transformer' in name and 'mlp' not in name:
                    # Extract CLS token (first token) from the output
                    # Output shape: [batch_size, num_patches + 1, dim]
                    # CLS token is at index 0
                    self.cls_token_embeddings = output[:, 0, :]  # [batch_size, dim]
                    
                    # Also capture all patch embeddings (excluding CLS token)
                    self.patch_embeddings = output[:, 1:, :]  # [batch_size, num_patches, dim]
                    
            return hook
        
        # Register hooks on transformer blocks
        for name, module in self.original_model.named_modules():
            if 'transformer' in name and 'mlp' not in name:
                module.register_forward_hook(get_cls_token_hook(name))
                print(f"Registered CLS token hook on layer: {name}")
                break
        
        # If no transformer hook found, try to register on the main transformer
        if not hasattr(self, '_hook_registered'):
            print("Trying to register hook on main transformer...")
            # Look for the main transformer module
            for name, module in self.original_model.named_modules():
                if 'transformer' in name and hasattr(module, 'layers'):
                    module.register_forward_hook(get_cls_token_hook(name))
                    print(f"Registered hook on main transformer: {name}")
                    break
    
    def forward(self, x):
        self.cls_token_embeddings = None
        self.patch_embeddings = None
        output = self.original_model(x)
        return output

def create_vit_model_for_feature_extraction():
    """Create a ViT model for feature extraction"""
    try:
        from vit_pytorch import ViT
        
        # Create a ViT model with the same architecture as your trained model
        model = ViT(
            image_size = 32,  # CIFAR-10 image size
            patch_size = 4,    # Based on your model name: patch4
            num_classes = 10,  # CIFAR-10 classes
            dim = 256,         # Based on your model name: dim256
            depth = 6,         # Based on your model name: depth6
            heads = 8,         # Based on your model name: heads8
            mlp_dim = 512,    # Based on your model name: mlp_dim
            channels = 3       # RGB images
        )
        
        print("Created ViT model for feature extraction")
        return model
        
    except ImportError:
        print("vit_pytorch not available, using placeholder")
        return None

def extract_vit_features(model, data_loader, device, max_samples=2000):
    """Extract CLS token embeddings from ViT model"""
    model.eval()
    all_cls_embeddings = []
    all_patch_embeddings = []
    all_labels = []
    
    print(f"Starting CLS token extraction, max samples: {max_samples}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * data.size(0) >= max_samples:
                break
                
            data, target = data.to(device), target.to(device)
            _ = model(data)  # Forward pass to trigger hooks
            
            if model.cls_token_embeddings is not None:
                cls_embeddings = model.cls_token_embeddings.cpu().numpy()
                all_cls_embeddings.append(cls_embeddings)
                all_labels.append(target.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")
            else:
                print(f"Warning: No CLS token embeddings captured in batch {batch_idx}")
    
    if not all_cls_embeddings:
        raise ValueError("No CLS token embeddings extracted. Please check if hooks are working properly.")
    
    # Concatenate all embeddings and labels
    all_cls_embeddings = np.vstack(all_cls_embeddings)
    all_labels = np.concatenate(all_labels)
    
    print(f"Extracted CLS token embeddings shape: {all_cls_embeddings.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return all_cls_embeddings, all_labels

def perform_pca_analysis(features, n_components=50):
    """Perform PCA analysis on CLS token embeddings"""
    print("\n" + "="*60)
    print("Starting PCA Analysis on CLS Token Embeddings")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform PCA
    n_components = min(n_components, features.shape[1])
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"Explained variance ratio for first 5 principal components:")
    for i in range(min(5, len(explained_variance_ratio))):
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]:.2%})")
    
    print(f"\nCumulative explained variance ratio:")
    for i in range(min(10, len(cumulative_variance_ratio))):
        print(f"  First {i+1} components: {cumulative_variance_ratio[i]:.4f} ({cumulative_variance_ratio[i]:.2%})")
    
    # Find number of components needed for 95% and 99% variance
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
    
    print(f"\nNumber of components needed for 95% variance: {n_components_95}")
    print(f"Number of components needed for 99% variance: {n_components_99}")
    
    return pca, features_pca, features_scaled, scaler

def plot_pca_results(pca, features_pca, labels, outdir, model_type):
    """Plot PCA analysis results for CLS token embeddings - Only keep 4 most important charts"""
    os.makedirs(outdir, exist_ok=True)
    
    # Set font for better visualization
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
    plt.title(f'{model_type} - CLS Token PCA\nExplained Variance Ratio')
    plt.grid(True, alpha=0.3)
    
    # 2. Cumulative variance ratio - Show how many components needed
    plt.subplot(2, 2, 2)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), 
             cumulative_variance, 'bo-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance', linewidth=2)
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'{model_type} - CLS Token PCA\nCumulative Explained Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Scatter plot of first 2 PCs - Show data distribution
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=30)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title(f'{model_type} - CLS Token PCA\nScatter Plot of First 2 PCs')
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    
    # 4. 3D Scatter plot of first 3 PCs - Show 3D distribution
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    scatter3d = ax3d.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], 
                             c=labels, alpha=0.7, cmap='tab10', s=20)
    ax3d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax3d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax3d.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax3d.set_title(f'{model_type} - CLS Token PCA\n3D Scatter Plot of First 3 PCs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{model_type}_cls_token_pca_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CLS Token PCA analysis plots saved to: {outdir}")
    print("Only kept 4 most important charts: explained variance ratio, cumulative variance ratio, first 2 PCs scatter plot, first 3 PCs 3D scatter plot")

def save_pca_results(pca, features_pca, labels, outdir, model_type):
    """Save PCA analysis results for CLS token embeddings"""
    os.makedirs(outdir, exist_ok=True)
    
    # Save PCA transformed features
    np.save(os.path.join(outdir, f'{model_type}_cls_token_features_pca.npy'), features_pca)
    
    # Save labels
    np.save(os.path.join(outdir, f'{model_type}_cls_token_labels.npy'), labels)
    
    # Save PCA components
    np.save(os.path.join(outdir, f'{model_type}_cls_token_pca_components.npy'), pca.components_)
    
    # Save explained variance ratio
    np.save(os.path.join(outdir, f'{model_type}_cls_token_explained_variance_ratio.npy'), 
            pca.explained_variance_ratio_)
    
    # Save as CSV format
    n_components = min(50, features_pca.shape[1])
    pca_df = pd.DataFrame(features_pca[:, :n_components], 
                          columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['class'] = labels
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    pca_df['class_name'] = np.array(class_names)[labels]
    pca_df.to_csv(os.path.join(outdir, f'{model_type}_cls_token_pca_results.csv'), index=False)
    
    # Save PCA statistics
    stats_df = pd.DataFrame({
        'PC': range(1, len(pca.explained_variance_ratio_) + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    stats_df.to_csv(os.path.join(outdir, f'{model_type}_cls_token_pca_stats.csv'), index=False)
    
    print(f"CLS token PCA analysis results saved to: {outdir}")

def main():
    """Main function for ViT CLS token PCA analysis"""
    # Configuration parameters
    model_type = "patch4_dim256_depth6_heads8"  # Based on your model name
    batch_size = 128
    max_samples = 2000  # Maximum number of samples
    outdir = f"runs/{model_type}_cls_token_pca"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create ViT model
    print(f"Creating ViT model for feature extraction...")
    model = create_vit_model_for_feature_extraction()
    if model is None:
        print("Failed to create ViT model")
        return
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT model created successfully:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Data transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # Load CIFAR-10 test dataset
    print("Loading CIFAR-10 test dataset...")
    test_dataset = datasets.CIFAR10('./data', train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Wrap model with hooks for CLS token extraction
    model_with_hooks = ViTCLSTokenExtractor(model)
    model_with_hooks = model_to_device(model_with_hooks, device)
    
    # Extract CLS token embeddings
    print("Starting CLS token embedding extraction...")
    features, labels = extract_vit_features(model_with_hooks, test_loader, device, max_samples)
    
    # Perform PCA analysis
    print("Starting PCA analysis on CLS token embeddings...")
    pca, features_pca, features_scaled, scaler = perform_pca_analysis(features, n_components=50)
    
    # Plot results
    print("Plotting PCA analysis results...")
    plot_pca_results(pca, features_pca, labels, outdir, model_type)
    
    # Save results
    print("Saving PCA analysis results...")
    save_pca_results(pca, features_pca, labels, outdir, model_type)
    
    print(f"\n" + "="*60)
    print("ViT CLS Token PCA Analysis Complete!")
    print("="*60)
    print(f"Total variance explained by first 10 PCs: {np.sum(pca.explained_variance_ratio_[:10]):.3f}")
    print(f"Results saved to: {outdir}")

def model_to_device(model, device):
    """Helper function to move model to device"""
    try:
        return model.to(device)
    except Exception as e:
        print(f"Warning: Could not move model to device {device}: {e}")
        return model

if __name__ == "__main__":
    main()
