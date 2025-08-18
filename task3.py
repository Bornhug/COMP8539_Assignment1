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
    Wrap vit_pytorch.ViT, inserting N learnable register tokens after [CLS].
    Registers participate in attention but are dropped at output.
    """
    def __init__(self, vit_model: ViT, num_registers: int = 4):
        super().__init__()
        assert isinstance(vit_model, ViT)
        self.vit = vit_model
        self.num_registers = int(num_registers)
        self.dim = vit_model.dim

        if self.num_registers > 0:
            # register tokens & their own (learnable) positional embeddings
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, self.dim))
            self.register_pos_embed = nn.Parameter(torch.zeros(1, self.num_registers, self.dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
            nn.init.trunc_normal_(self.register_pos_embed, std=0.02)

    def forward(self, x, return_tokens=False):
        vit = self.vit
        b = x.shape[0]

        # ---- patch embedding (B, N, D)
        x = vit.to_patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # ---- tokens: [CLS] + [REG]*R + patches
        cls = vit.cls_token.expand(b, -1, -1)
        if self.num_registers > 0:
            regs = self.register_tokens.expand(b, -1, -1)
            tokens = torch.cat([cls, regs, x], dim=1)
        else:
            tokens = torch.cat([cls, x], dim=1)

        # ---- positional embeddings
        if self.num_registers > 0:
            cls_pos   = vit.pos_embedding[:, :1, :]
            patch_pos = vit.pos_embedding[:, 1:1 + x.size(1), :]
            pos = torch.cat([cls_pos, self.register_pos_embed, patch_pos], dim=1)
        else:
            pos = vit.pos_embedding[:, :tokens.size(1), :]

        x = tokens + pos
        x = vit.dropout(x)

        # ---- transformer blocks
        x = vit.transformer(x)

        # ---- separate cls and patch tokens
        cls_out = x[:, 0]
        patch_tokens = x[:, (1 + self.num_registers):, :]   # (B, N, D)

        logits = vit.mlp_head(cls_out)

        if return_tokens:
            return logits, patch_tokens
        return logits


# ───────────────────────── outlier ratio function ─────────────────────────
def compute_outlier_ratio(patch_tokens, threshold=150, top_percent=0.02):
    """
    Compute outlier ratio of patch embeddings.
    - threshold: fixed L2 norm cutoff (e.g., >150)
    - top_percent: proportion of largest norms (e.g., top 2%)
    """
    norms = patch_tokens.norm(dim=-1).view(-1)  # flatten all tokens
    ratio_thresh = (norms > threshold).float().mean().item()

    q = torch.quantile(norms, 1 - top_percent)
    ratio_top = (norms > q).float().mean().item()

    return ratio_thresh, ratio_top


# ───────────────────────── evaluation ─────────────────────────
def evaluate(model, loader, device, criterion):
    """Run evaluation on test set, return accuracy, loss, and outlier ratio."""
    model.eval()
    correct, running_loss = 0, 0.0
    ratio_thresh_list, ratio_top_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, patch_tokens = model(x, return_tokens=True)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

            # compute outlier ratio
            r_thresh, r_top = compute_outlier_ratio(patch_tokens)
            ratio_thresh_list.append(r_thresh)
            ratio_top_list.append(r_top)

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    avg_ratio_thresh = sum(ratio_thresh_list) / len(ratio_thresh_list)
    avg_ratio_top = sum(ratio_top_list) / len(ratio_top_list)

    return accuracy, avg_loss, avg_ratio_thresh, avg_ratio_top


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
    # ───────────────────── output directory ─────────────────────
    outdir = f"runs/patch{args.patch_size}_dim{args.dim}_depth{args.depth}_heads{args.heads}"
    os.makedirs(outdir, exist_ok=True)
    print("Saving artifacts to:", outdir)

    # ───────────────────────── device info ──────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"→ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("→ CUDA not available – falling back to CPU")

    # ───────────────────────── model ────────────────────────────
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

    # wrap with register tokens
    if args.registers > 0:
        model = ViTWithRegisters(base_vit, num_registers=args.registers).to(device)
    else:
        model = base_vit.to(device)

    # ───────────────────────── data ─────────────────────────────
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
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ───────────────────────── optimiser ────────────────────────
    optimiser = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
    criterion  = nn.CrossEntropyLoss()

    # ───────────────────────── logging ──────────────────────────
    train_loss_hist, train_acc_hist = [], []
    test_loss_hist,  test_acc_hist  = [], []
    ratio_thresh_hist, ratio_top_hist = [], []

    csv_path = os.path.join(outdir, "metrics.csv")
    csv_log  = open(csv_path, "w", newline='')
    csv_writer = csv.writer(csv_log)
    csv_writer.writerow(["epoch", "train_loss", "train_acc",
                         "test_acc", "test_loss",
                         "outlier_ratio_thresh", "outlier_ratio_top",
                         "epoch_time_sec", "peak_mem_MB"])

    overall_start = time.time()
    peak_mem_global = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # ───────── training ─────────
        model.train()
        running_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            logits, _ = model(x, return_tokens=True)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        train_loss = running_loss / len(train_ds)
        train_acc  = correct / len(train_ds)

        # ───────── evaluation ───────
        test_acc, test_loss, ratio_thresh, ratio_top = evaluate(model, test_loader, device, criterion)

        # ───────── bookkeeping ──────
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
        ratio_thresh_hist.append(ratio_thresh)
        ratio_top_hist.append(ratio_top)

        csv_writer.writerow([epoch, train_loss, train_acc,
                             test_acc, test_loss,
                             ratio_thresh, ratio_top,
                             elapsed, mem_mb])

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | "
              f"train_acc={train_acc*100:6.2f}% | "
              f"test_loss={test_loss:.4f} | "
              f"test_acc={test_acc*100:6.2f}% | "
              f"outlier>150={ratio_thresh*100:5.2f}% | "
              f"outlier_top2%={ratio_top*100:5.2f}% | "
              f"epoch_time={elapsed:5.1f}s | "
              f"peak_mem={mem_mb:7.1f} MB")

    # ───────────────────────── final report ─────────────────────
    total_time = time.time() - overall_start
    final_test_acc, final_test_loss, final_ratio_thresh, final_ratio_top = evaluate(model, test_loader, device, criterion)

    print("\n──────────── Final Results ────────────")
    print(f"Best test acc : {max(test_acc_hist)*100:6.2f}%")
    print(f"Final test acc: {final_test_acc*100:6.2f}%")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final outlier>150: {final_ratio_thresh*100:.2f}%")
    print(f"Final outlier top-2%: {final_ratio_top*100:.2f}%")
    print(f"Total runtime : {total_time/60:.1f} min")
    if torch.cuda.is_available():
        print(f"Peak GPU mem  : {peak_mem_global:.1f} MB")

    # save artifacts ------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "session_stats.json"), "w") as f:
        json.dump({"total_runtime_min": round(total_time/60, 2),
                   "overall_peak_gpu_mem_MB": round(peak_mem_global, 1),
                   "final_test_loss": round(final_test_loss, 4),
                   "final_test_acc":  round(final_test_acc*100, 2),
                   "final_outlier_ratio_thresh": round(final_ratio_thresh, 4),
                   "final_outlier_ratio_top": round(final_ratio_top, 4)},
                  f, indent=2)
    csv_log.close()

    # ─────────── save curves ────────────
    epochs = range(1, args.epochs + 1)
    plot_metric(outdir, epochs, train_loss_hist, test_loss_hist, "Loss", "loss_curve.png")
    plot_metric(outdir, epochs, train_acc_hist,  test_acc_hist,  "Accuracy", "accuracy_curve.png")
    print("Artifacts stored in", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--dim",        type=int, default=256)
    parser.add_argument("--depth",      type=int, default=6)
    parser.add_argument("--heads",      type=int, default=8)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--registers", type=int, default=4, help="number of register tokens (0 to disable)")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)
    main(args)
