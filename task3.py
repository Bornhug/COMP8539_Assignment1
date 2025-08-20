# task3.py
import time, csv, argparse, math, os, json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vit_pytorch import ViT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ViTWithRegisters(nn.Module):
    """
    Wrap vit_pytorch.ViT, inserting R learnable register tokens after [CLS].
    Registers participate in attention but are dropped at output.
    Supports toggling registers ON/OFF at inference for virtual ablation.
    """
    def __init__(self, vit_model: ViT, num_registers: int = 4):
        super().__init__()
        assert isinstance(vit_model, ViT)
        self.vit = vit_model
        self.num_registers = int(num_registers)

        # Robustly infer embedding dim (different vit_pytorch versions)
        if hasattr(vit_model, "pos_embedding"):
            self.dim = vit_model.pos_embedding.shape[-1]
        else:
            self.dim = vit_model.cls_token.shape[-1]

        if self.num_registers > 0:
            # Register tokens and their learnable positional embeddings
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, self.dim))
            self.register_pos_embed = nn.Parameter(torch.zeros(1, self.num_registers, self.dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
            nn.init.trunc_normal_(self.register_pos_embed, std=0.02)

    def forward(self, x, return_tokens=False, use_registers=True):
        vit = self.vit
        b = x.shape[0]

        # ---- patch embedding -> (B, N, D)
        x = vit.to_patch_embedding(x)
        # Some versions return (B, D, H', W'); flatten if needed
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        patch_len = x.size(1)

        # ---- tokens: [CLS] + (optional REGs) + patches
        cls = vit.cls_token.expand(b, -1, -1)  # (B, 1, D)
        if self.num_registers > 0 and use_registers:
            regs = self.register_tokens.expand(b, -1, -1)  # (B, R, D)
            tokens = torch.cat([cls, regs, x], dim=1)      # (B, 1+R+N, D)
        else:
            tokens = torch.cat([cls, x], dim=1)            # (B, 1+N, D)

        # ---- positional embeddings
        # vit.pos_embedding is typically (1, 1+N_pos, D)
        cls_pos   = vit.pos_embedding[:, :1, :]
        patch_pos = vit.pos_embedding[:, 1:1 + patch_len, :]
        if self.num_registers > 0 and use_registers:
            pos = torch.cat([cls_pos, self.register_pos_embed, patch_pos], dim=1)
        else:
            pos = torch.cat([cls_pos, patch_pos], dim=1)

        x = tokens + pos
        x = vit.dropout(x)

        # ---- transformer blocks
        x = vit.transformer(x)  # (B, 1(+R)+N, D)

        # ---- outputs
        cls_out = x[:, 0]
        # Patch tokens exclude [CLS] and (optionally) registers
        offset = 1 + (self.num_registers if (self.num_registers > 0 and use_registers) else 1)
        # Note: when no registers are used, offset=2 would be wrong; we want 1.
        # Fix offset:
        offset = 1 + (self.num_registers if (self.num_registers > 0 and use_registers) else 0)
        patch_tokens = x[:, offset:, :]  # (B, N, D)

        logits = vit.mlp_head(cls_out)
        if return_tokens:
            return logits, patch_tokens
        return logits


# ----------------------------- outlier ratio -----------------------------
def compute_outlier_ratio(patch_tokens, threshold=150, top_percent=0.02):
    """
    Compute outlier ratio of patch embeddings.
    - threshold: fixed L2 norm cutoff (e.g., >150)
    - top_percent: proportion of largest norms (e.g., top 2%)
    """
    norms = patch_tokens.norm(dim=-1).reshape(-1)  # flatten all tokens
    ratio_thresh = (norms > threshold).float().mean().item()
    q = torch.quantile(norms, 1 - top_percent)
    ratio_top = (norms > q).float().mean().item()
    return ratio_thresh, ratio_top


# ------------------------------- evaluate -------------------------------
def evaluate(model, loader, device, criterion):
    """
    Return:
      acc, loss,
      outlier(with-reg)>150, outlier(with-reg)top2%,
      outlier(no-reg)>150,  outlier(no-reg)top2%.
    """
    model.eval()
    correct, running_loss = 0, 0.0
    r_thr_w, r_top_w, r_thr_n, r_top_n = [], [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # WITH registers
            logits_w, patches_w = model(x, return_tokens=True, use_registers=True)
            loss = criterion(logits_w, y)
            running_loss += loss.item() * x.size(0)
            correct += (logits_w.argmax(1) == y).sum().item()
            a, b = compute_outlier_ratio(patches_w)
            r_thr_w.append(a); r_top_w.append(b)

            # NO registers (virtual ablation baseline)
            _, patches_n = model(x, return_tokens=True, use_registers=False)
            c, d = compute_outlier_ratio(patches_n)
            r_thr_n.append(c); r_top_n.append(d)

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return (accuracy, avg_loss,
            sum(r_thr_w)/len(r_thr_w), sum(r_top_w)/len(r_top_w),
            sum(r_thr_n)/len(r_thr_n), sum(r_top_n)/len(r_top_n))


def plot_metric(outdir: str, epochs, train_values, test_values, ylabel: str, filename: str):
    """Plot train/test curves and save to a file."""
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_values, label=f"train_{ylabel.lower()}")
    plt.plot(epochs, test_values,  label=f"test_{ylabel.lower()}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"ViT • CIFAR-10 • {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300)
    plt.close()


def main(args):
    # ------------------------- output directory -------------------------
    outdir = f"runs/patch{args.patch_size}_dim{args.dim}_depth{args.depth}_heads{args.heads}"
    os.makedirs(outdir, exist_ok=True)
    print("Saving artifacts to:", outdir)

    # ----------------------------- device -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"→ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("→ CUDA not available – falling back to CPU")

    # ------------------------------ model -------------------------------
    base_vit = ViT(
        image_size=32,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.dim * 2,
        dropout=0.1,
        emb_dropout=0.1
    )
    # Always wrap so forward signature is unified
    model = ViTWithRegisters(base_vit, num_registers=args.registers).to(device)

    # ------------------------------- data -------------------------------
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---------------------------- optimizer -----------------------------
    optimiser = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
    criterion  = nn.CrossEntropyLoss()

    # ----------------------------- logging ------------------------------
    train_loss_hist, train_acc_hist = [], []
    test_loss_hist,  test_acc_hist  = [], []
    w_thr_hist, w_top_hist = [], []
    n_thr_hist, n_top_hist = [], []

    csv_path = os.path.join(outdir, "metrics.csv")
    csv_log  = open(csv_path, "w", newline='')
    csv_writer = csv.writer(csv_log)
    csv_writer.writerow(["epoch", "train_loss", "train_acc",
                         "test_acc", "test_loss",
                         "outlier_w>150", "outlier_w_top2%",
                         "outlier_no>150", "outlier_no_top2%",
                         "epoch_time_sec", "peak_mem_MB"])

    overall_start = time.time()
    peak_mem_global = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # ---------------------------- train -----------------------------
        model.train()
        running_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            logits, _ = model(x, return_tokens=True, use_registers=True)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        train_loss = running_loss / len(train_ds)
        train_acc  = correct / len(train_ds)

        # --------------------------- evaluate ---------------------------
        (test_acc, test_loss,
         out_w_thr, out_w_top,
         out_n_thr, out_n_top) = evaluate(model, test_loader, device, criterion)

        # -------------------------- bookkeeping -------------------------
        elapsed = time.time() - epoch_start
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mem_global = max(peak_mem_global, mem_mb)
        else:
            mem_mb = math.nan

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        w_thr_hist.append(out_w_thr); w_top_hist.append(out_w_top)
        n_thr_hist.append(out_n_thr); n_top_hist.append(out_n_top)

        csv_writer.writerow([epoch, train_loss, train_acc,
                             test_acc, test_loss,
                             out_w_thr, out_w_top,
                             out_n_thr, out_n_top,
                             elapsed, mem_mb])

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | train_acc={train_acc*100:6.2f}% | "
              f"test_loss={test_loss:.4f} | test_acc={test_acc*100:6.2f}% | "
              f"wREG>150={out_w_thr*100:5.2f}% | wREGtop2%={out_w_top*100:5.2f}% | "
              f"noREG>150={out_n_thr*100:5.2f}% | noREGtop2%={out_n_top*100:5.2f}% | "
              f"epoch_time={elapsed:5.1f}s | peak_mem={mem_mb:7.1f} MB")

    # --------------------------- final report ---------------------------
    total_time = time.time() - overall_start
    (final_acc, final_loss,
     final_w_thr, final_w_top,
     final_n_thr, final_n_top) = evaluate(model, test_loader, device, criterion)

    print("\n──────────── Final Results ────────────")
    print(f"Best test acc        : {max(test_acc_hist)*100:6.2f}%")
    print(f"Final test acc       : {final_acc*100:6.2f}%")
    print(f"Final test loss      : {final_loss:.4f}")
    print(f"Final outlier >150   : {final_w_thr*100:.2f}% (with_reg) / {final_n_thr*100:.2f}% (no_reg)")
    print(f"Final outlier top-2% : {final_w_top*100:.2f}% (with_reg) / {final_n_top*100:.2f}% (no_reg)")
    print(f"Total runtime        : {total_time/60:.1f} min")
    if torch.cuda.is_available():
        print(f"Peak GPU mem         : {peak_mem_global:.1f} MB")

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "session_stats.json"), "w") as f:
        json.dump({
            "total_runtime_min": round(total_time/60, 2),
            "overall_peak_gpu_mem_MB": round(peak_mem_global, 1),
            "final_test_loss": round(final_loss, 4),
            "final_test_acc":  round(final_acc*100, 2),
            "final_outlier_w>150": round(final_w_thr, 4),
            "final_outlier_w_top2%": round(final_w_top, 4),
            "final_outlier_no>150": round(final_n_thr, 4),
            "final_outlier_no_top2%": round(final_n_top, 4),
        }, f, indent=2)
    csv_log.close()

    # Curves
    epochs = range(1, args.epochs + 1)
    plot_metric(outdir, epochs, train_loss_hist, test_loss_hist, "Loss", "loss_curve.png")
    plot_metric(outdir, epochs, train_acc_hist,  test_acc_hist,  "Accuracy", "accuracy_curve.png")
    print("Artifacts stored in", outdir)


def analyze_register_effect(model, test_loader, device, criterion):
    """
    Analyze how register tokens affect outlier ratios on the same trained model.
    Runs two forwards per batch: with registers ON and OFF.
    """
    print("\n" + "="*60)
    print("REGISTER TOKENS EFFECT ANALYSIS")
    print("="*60)

    model.eval()
    total_samples = 0
    correct_with_reg = 0
    correct_no_reg = 0
    ratios_with_reg = {"thresh": [], "top": []}
    ratios_no_reg   = {"thresh": [], "top": []}

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = x.size(0)
            total_samples += bs

            # With registers
            logits_w, patches_w = model(x, return_tokens=True, use_registers=True)
            correct_with_reg += (logits_w.argmax(1) == y).sum().item()
            a, b = compute_outlier_ratio(patches_w)
            ratios_with_reg["thresh"].append(a)
            ratios_with_reg["top"].append(b)

            # No registers
            logits_n, patches_n = model(x, return_tokens=True, use_registers=False)
            correct_no_reg += (logits_n.argmax(1) == y).sum().item()
            c, d = compute_outlier_ratio(patches_n)
            ratios_no_reg["thresh"].append(c)
            ratios_no_reg["top"].append(d)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx+1}/{len(test_loader)}")

    avg_w_thr = sum(ratios_with_reg["thresh"]) / len(ratios_with_reg["thresh"])
    avg_w_top = sum(ratios_with_reg["top"])   / len(ratios_with_reg["top"])
    avg_n_thr = sum(ratios_no_reg["thresh"])  / len(ratios_no_reg["thresh"])
    avg_n_top = sum(ratios_no_reg["top"])     / len(ratios_no_reg["top"])

    acc_w = correct_with_reg / total_samples
    acc_n = correct_no_reg   / total_samples

    print(f"\nResults Summary:")
    print(f"Total samples                : {total_samples}")
    print(f"Accuracy with registers      : {acc_w*100:.2f}%")
    print(f"Accuracy without registers   : {acc_n*100:.2f}%")
    print(f"Accuracy difference          : {(acc_w - acc_n)*100:+.2f}%")
    print(f"\nOutlier Ratio (fixed >150)   : {avg_w_thr*100:.2f}% (with) / {avg_n_thr*100:.2f}% (no)")
    print(f"Outlier Ratio (top 2%)       : {avg_w_top*100:.2f}% (with) / {avg_n_top*100:.2f}% (no)")

    return {
        "accuracy_with_reg": acc_w,
        "accuracy_no_reg": acc_n,
        "accuracy_difference": acc_w - acc_n,
        "outlier_thresh_with_reg": avg_w_thr,
        "outlier_thresh_no_reg": avg_n_thr,
        "outlier_thresh_difference": avg_w_thr - avg_n_thr,
        "outlier_top_with_reg": avg_w_top,
        "outlier_top_no_reg": avg_n_top,
        "outlier_top_difference": avg_w_top - avg_n_top
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--dim",        type=int, default=256)
    parser.add_argument("--depth",      type=int, default=6)
    parser.add_argument("--heads",      type=int, default=8)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--registers",  type=int, default=4,
                        help="number of register tokens (0 to disable)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="only run register effect analysis on existing model")

    args = parser.parse_args()

    # Colab/Jupyter-friendly start method
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.analyze_only:
        # Load existing model and run analysis only
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outdir = f"runs/patch{args.patch_size}_dim{args.dim}_depth{args.depth}_heads{args.heads}"

        # Build and wrap model (always wrapped for unified forward)
        base_vit = ViT(
            image_size=32,
            patch_size=args.patch_size,
            num_classes=10,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.dim * 2,
            dropout=0.1,
            emb_dropout=0.1
        )
        model = ViTWithRegisters(base_vit, num_registers=args.registers).to(device)

        # Load weights
        model.load_state_dict(torch.load(os.path.join(outdir, "model.pt"), map_location=device))

        # Test loader
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                                 num_workers=2, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        results = analyze_register_effect(model, test_loader, device, criterion)

        # Save analysis
        analysis_file = os.path.join(outdir, "register_effect_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis results saved to: {analysis_file}")
    else:
        # Full training run
        main(args)
