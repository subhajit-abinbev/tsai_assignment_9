import argparse, os, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import RandAugment
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import boto3
from botocore.exceptions import ClientError
from datasets import load_from_disk

# ==========================================================
# 0. S3 Utilities
# ==========================================================
def upload_to_s3(local_path, bucket, s3_path):
    """Upload file to S3"""
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, s3_path)
        print(f"✅ Uploaded {local_path} → s3://{bucket}/{s3_path}")
    except ClientError as e:
        print(f"⚠️ Failed to upload {local_path}: {e}")

def download_from_s3(bucket, s3_path, local_path):
    """Download file from S3 if exists"""
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, s3_path, local_path)
        print(f"✅ Downloaded checkpoint from s3://{bucket}/{s3_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("No checkpoint found on S3.")
        else:
            print(f"⚠️ Failed to download checkpoint: {e}")
        return False


# ==========================================================
# 1.  ResNet-50 architecture (from scratch)
# ==========================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),  # Using SiLU instead of ReLU
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w
    
def get_norm_layer(num_features, distributed=False):
    """Return appropriate normalization layer"""
    if distributed:
        return nn.SyncBatchNorm(num_features)
    else:
        return nn.BatchNorm2d(num_features)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, use_se=True, distributed=False):
        super().__init__()
        
        # ResNet-D: Move stride to 3x3 conv for better downsampling
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = get_norm_layer(planes, distributed)
        
        # 3x3 conv gets the stride (ResNet-D trick)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm_layer(planes, distributed)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm_layer(planes * self.expansion, distributed)

        # Use SiLU instead of ReLU
        self.silu = nn.SiLU(inplace=True)
        
        # SE Block
        self.se = SEBlock(planes * self.expansion) if use_se else nn.Identity()
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.silu(self.bn1(self.conv1(x)))
        out = self.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Apply SE attention
        out = self.se(out)
        
        out += identity
        out = self.silu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, distributed=False):
        super().__init__()
        self.inplanes = 64
        self.distributed = distributed
        
        # ResNet-D Stem: 3 conv layers instead of 7x7
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = get_norm_layer(32, distributed)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm_layer(32, distributed)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = get_norm_layer(64, distributed)

        self.silu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # ResNet-D downsampling: avgpool + 1x1 conv when stride > 1
            downsample_layers = []
            if stride > 1:
                downsample_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
            downsample_layers.extend([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=1, bias=False),  # Always stride=1 after avgpool
                get_norm_layer(planes * block.expansion, self.distributed),
            ])
            downsample = nn.Sequential(*downsample_layers)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use 'linear' for SiLU since there's no direct SiLU option
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch (ResNet trick)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        # ResNet-D stem
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.silu(self.bn2(self.conv2(x)))
        x = self.silu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def build_model(num_classes, dataset="imagenet", distributed=False):
    """Build enhanced ResNet-50 with all improvements"""
    if dataset.lower() == "cifar100":
        # For CIFAR-100, use smaller stem
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, distributed)
        # Modify for CIFAR (no stride in first conv, no maxpool)
        model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model._init_weights()
        return model
    else:
        # Full ResNet-D for ImageNet
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, distributed)

# ==========================================================
# 2.  Data utilities
# ==========================================================
# Add these imports at the top
def get_strong_transforms():
    """Enhanced augmentation for better accuracy"""
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),  # Add random erasing
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tr_tf

def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_data(args):
    """Return train and val loaders for CIFAR100, ImageNet-100, or ImageNet."""
    if args.dataset.lower() == "cifar100":
        num_classes = 100
        # CIFAR-100 specific transforms (32x32 images)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409],  # CIFAR-100 specific stats
                std=[0.2673, 0.2564, 0.2762]
            ),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2673, 0.2564, 0.2762]
            ),
        ])
        train_dataset = datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=val_transform
        )
    elif args.dataset.lower() in ["imagenet100", "imagenet"]:
        # Use HuggingFace dataset
        from torch.utils.data import Dataset
        
        class HFImageNetDataset(Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.dataset = hf_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                sample = self.dataset[idx]
                image = sample['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                label = sample['label']
                
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        print(f"Loading HuggingFace dataset from: {args.data_dir}")
        dataset = load_from_disk(args.data_dir)
        
        num_classes = 1000 if args.dataset.lower() == "imagenet" else 100
        
        train_transform = get_strong_transforms()
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        train_dataset = HFImageNetDataset(dataset['train'], train_transform)
        val_split = 'validation' if 'validation' in dataset else 'val'
        val_dataset = HFImageNetDataset(dataset[val_split], val_transform)
    else:
        raise ValueError("Unsupported dataset: choose cifar100, imagenet100, or imagenet")

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = distributed.DistributedSampler(train_dataset)
        val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, num_classes

# ==========================================================
# 3.  LR Finder
# ==========================================================
def lr_finder(model, loader, criterion, optimizer, device,
              start_lr=1e-5, end_lr=1, num_iters=100, beta=0.98):
    """Exponential LR range test (loss vs LR)."""
    mult = (end_lr / start_lr) ** (1 / num_iters)
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss, best_loss = 0., 1e9
    losses, lrs = [], []
    scaler = GradScaler()
    model.train()

    for batch_num, (inputs, targets) in enumerate(loader):
        if batch_num > num_iters:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed = avg_loss / (1 - beta ** (batch_num + 1))

        if smoothed < best_loss or batch_num == 0:
            best_loss = smoothed
        if smoothed > 4 * best_loss:
            break

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(smoothed)
        lrs.append(lr)
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    print("LR Finder complete. Plot loss vs lr to pick best max_lr.")
    return lrs, losses

# ==========================================================
# 4.  Training / evaluation
# ==========================================================
class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def save_checkpoint(state, path, best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if best:
        torch.save(state, os.path.join(os.path.dirname(path), "model_best.pth"))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
def train(rank, world_size, args):
    if args.distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Initialize writer early to avoid UnboundLocalError
    writer = SummaryWriter(args.log_dir) if rank == 0 else None

    try:
        # ===============================
        # 1. Data
        # ===============================
        train_loader, val_loader, num_classes = get_data(args)

        # ===============================
        # 2. Model
        # ===============================
        model = build_model(num_classes, args.dataset, args.distributed).to(device)
        if args.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

        # ===============================
        # 3. EMA
        # ===============================
        ema_model = model.module if args.distributed else model
        ema = EMA(ema_model, decay=0.9999)

        # ===============================
        # 4. Criterion and Optimizer
        # ===============================
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, epochs=args.epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.1,
            anneal_strategy='cos', div_factor=10.0, final_div_factor=1e3
        )

        scaler = GradScaler()
        best_acc, start_epoch = 0.0, 0
        writer = SummaryWriter(args.log_dir) if rank == 0 else None

        # ===============================
        # 5. LR Finder Mode
        # ===============================
        if args.find_lr:
            lrs, losses = lr_finder(model, train_loader, criterion, optimizer, device)
            if rank == 0:
                np.savez("lr_finder_results.npz", lrs=lrs, losses=losses)
                print("Saved lr_finder_results.npz (plot loss vs lr).")
            return

        # ===============================
        # 6. Resume Checkpoint
        # ===============================
        local_ckpt_dir = args.output_dir
        os.makedirs(local_ckpt_dir, exist_ok=True)

        if args.resume and args.s3_bucket:
            s3 = boto3.client('s3')
            response = s3.list_objects_v2(Bucket=args.s3_bucket, Prefix=args.s3_prefix)
            if 'Contents' in response:
                checkpoints = sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(".pth")])
                if checkpoints:
                    latest_ckpt = checkpoints[-1]
                    local_ckpt_path = os.path.join(local_ckpt_dir, os.path.basename(latest_ckpt))
                    download_from_s3(args.s3_bucket, latest_ckpt, local_ckpt_path)
                    checkpoint = torch.load(local_ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint["model"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    scaler.load_state_dict(checkpoint["scaler"])
                    ema.shadow = checkpoint["ema"]
                    best_acc = checkpoint.get("best_acc", 0.0)
                    start_epoch = checkpoint["epoch"] + 1
                    if rank == 0:
                        print(f"Resumed from checkpoint {latest_ckpt}")

        # ===============================
        # 7. Training Loop
        # ===============================
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, disable=(rank != 0))

            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(device), lbls.to(device)

                # --- MixUp 50% probability ---
                use_mixup = np.random.rand() < 0.5
                if use_mixup:
                    imgs, lbls_a, lbls_b, lam = mixup_data(imgs, lbls, alpha=0.2)
                    optimizer.zero_grad()
                    with autocast():
                        out = model(imgs)
                        loss = mixup_criterion(criterion, out, lbls_a, lbls_b, lam)
                else:
                    optimizer.zero_grad()
                    with autocast():
                        out = model(imgs)
                        loss = criterion(out, lbls)

                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(model.module if args.distributed else model)

                running_loss += loss.item()
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_description(f"Epoch[{epoch+1}/{args.epochs}] Loss {loss.item():.4f} LR {current_lr:.6f}")

            # ============ Validation & Checkpointing ============
            if rank == 0:
                model_eval = model.module if args.distributed else model
                ema.apply_shadow(model_eval)
                val_acc = evaluate(model_eval, val_loader, device)
                ema.restore(model_eval)

                avg_loss = running_loss / len(train_loader)
                print(f"Epoch {epoch+1}: Top1_Acc={val_acc:.2f}% Loss={avg_loss:.4f}")

                if writer:
                    writer.add_scalar("Loss/train", avg_loss, epoch)
                    writer.add_scalar("Acc/val_top1", val_acc, epoch)
                    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

                # Save checkpoint
                is_best = val_acc > best_acc
                best_acc = max(best_acc, val_acc)
                ckpt_path = os.path.join(args.output_dir, f"ckpt_{epoch}.pth")
                save_checkpoint({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                    "ema": ema.shadow,
                }, ckpt_path, best=is_best)

                # S3 Upload
                if args.s3_bucket:
                    upload_to_s3(ckpt_path, args.s3_bucket, f"{args.s3_prefix}/ckpt_{epoch}.pth")
                    if is_best:
                        upload_to_s3(os.path.join(args.output_dir, "model_best.pth"), args.s3_bucket,
                                     f"{args.s3_prefix}/model_best.pth")
    finally:
        if writer:
            writer.close()
        if args.distributed:
            dist.destroy_process_group()


# ==========================================================
# 5.  Main
# ==========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet100", "imagenet"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max-lr", type=float, default=0.2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--log-dir", default="./runs")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--find-lr", action="store_true")
    p.add_argument("--s3-bucket", type=str, help="S3 bucket to store checkpoints")
    p.add_argument("--s3-prefix", type=str, default="checkpoints", help="S3 prefix/path inside the bucket")
    p.add_argument("--resume", action="store_true", help="Resume training from latest S3 checkpoint")
    args = p.parse_args()

    world_size = torch.cuda.device_count()
    if args.distributed and world_size > 1:
        mp.spawn(train, nprocs=world_size, args=(world_size, args))
    else:
        train(0, 1, args)

if __name__ == "__main__":
    main()
