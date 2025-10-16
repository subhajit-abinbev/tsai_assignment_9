# ResNet-50 ImageNet Training from Scratch

**Goal**: Achieve 78%+ top-1 accuracy on ImageNet-1K using ResNet-50 trained from scratch with modern techniques.

## 🎯 **Key Features**

- **Enhanced ResNet-50 Architecture** with ResNet-D improvements
- **Squeeze-and-Excitation (SE) blocks** for attention mechanism
- **Advanced data augmentation** (RandAugment, MixUp, RandomErasing)
- **Modern training techniques** (Label Smoothing, EMA, OneCycleLR)
- **Distributed training support** for multi-GPU setups
- **Learning rate finder** for optimal hyperparameter selection
- **Mixed precision training** for faster training and memory efficiency

## 🏗️ **Architecture Improvements**

### ResNet-D Enhancements
- **3-conv stem** instead of 7×7 conv for better feature preservation
- **Improved downsampling** with AvgPool + 1×1 conv pattern
- **SiLU activation** for smoother gradient flow
- **Conditional BatchNorm** (SyncBN for distributed, regular BN for single GPU)

### Attention Mechanism
- **SE blocks** after each bottleneck for channel attention
- **16x reduction ratio** for optimal performance/efficiency trade-off

### Training Optimizations
- **Label smoothing** (0.1) to prevent overconfidence
- **MixUp augmentation** (α=0.2) for better generalization
- **Exponential Moving Average** (EMA) of model weights
- **Gradient clipping** for training stability

## 📋 **Requirements**

```bash
pip install torch torchvision tqdm tensorboard numpy matplotlib
```

## 🚀 **Quick Start**

### 1. **Find Optimal Learning Rate**
```bash
python train.py --dataset imagenet100 --data-dir ./data --find-lr --batch-size 128 --distributed
```

### 2. **Train on ImageNet-100 (Testing)**
```bash
python train.py --dataset imagenet100 --data-dir ./data --epochs 90 --batch-size 128 --max-lr 0.15 --distributed
```

### 3. **Scale to ImageNet-1K (Production)**
```bash
python train.py --dataset imagenet --data-dir ./imagenet_data --epochs 120 --batch-size 128 --max-lr 0.1 --distributed
```

## 📊 **Command Line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Required | Path to dataset directory |
| `--dataset` | `cifar100` | Dataset choice: `cifar100`, `imagenet100`, `imagenet` |
| `--epochs` | `120` | Number of training epochs |
| `--batch-size` | `128` | Batch size per GPU |
| `--lr` | `0.1` | Base learning rate |
| `--max-lr` | `0.2` | Maximum learning rate for OneCycleLR |
| `--momentum` | `0.9` | SGD momentum |
| `--weight-decay` | `1e-4` | Weight decay coefficient |
| `--workers` | `8` | Number of data loading workers |
| `--output-dir` | `./checkpoints` | Directory to save checkpoints |
| `--log-dir` | `./runs` | Directory for TensorBoard logs |
| `--distributed` | `False` | Enable distributed training |
| `--find-lr` | `False` | Run learning rate finder |

## 📁 **Data Directory Structure**

### ImageNet-1K / ImageNet-100
```
data_dir/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### CIFAR-100
```
data_dir/
└── cifar-100-python/  # Auto-downloaded
```

## 🔧 **Training Workflow**

### Step 1: Learning Rate Finding
```bash
# Find optimal learning rate
python train.py --dataset imagenet100 --data-dir ./data --find-lr --batch-size 128 --distributed

# This saves lr_finder_results.npz - plot loss vs LR to find optimal max_lr
```

### Step 2: Training on ImageNet-100
```bash
# Train on subset first to validate approach
python train.py \
    --dataset imagenet100 \
    --data-dir ./data \
    --epochs 90 \
    --batch-size 128 \
    --max-lr 0.15 \
    --distributed \
    --output-dir ./checkpoints_in100 \
    --log-dir ./runs_in100
```

### Step 3: Scale to ImageNet-1K
```bash
# Full ImageNet training
python train.py \
    --dataset imagenet \
    --data-dir ./imagenet_data \
    --epochs 120 \
    --batch-size 128 \
    --max-lr 0.1 \
    --distributed \
    --output-dir ./checkpoints_in1k \
    --log-dir ./runs_in1k
```

## ☁️ **AWS EC2 Setup Guide**

### 1. **Launch EC2 Instance**

**Recommended Instance: `g4dn.8xlarge`**
- **2x NVIDIA T4 GPUs** (16GB each)
- **32 vCPUs, 128GB RAM**
- **Perfect for distributed training**

**AMI Selection:**
- Use **Deep Learning AMI (Ubuntu 20.04)** - comes with PyTorch pre-installed
- Or use **Ubuntu 20.04 LTS** and install dependencies manually

### 2. **Instance Configuration**
```bash
# Storage: 500GB EBS GP3 (for ImageNet dataset)
# Security Group: Allow SSH (port 22) from your IP
# Key Pair: Create or use existing key pair for SSH access
```

### 3. **Connect to Instance**
```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com

# Or use AWS Session Manager for browser-based access
```

### 4. **Environment Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies (if not using Deep Learning AMI)
sudo apt install -y python3-pip git htop nvtop

# Install Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tqdm tensorboard numpy matplotlib

# Verify GPU access
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 5. **Clone Repository**
```bash
# Clone your repository
git clone https://github.com/your-username/tsai_assignment_9.git
cd tsai_assignment_9
```

### 6. **Download ImageNet Data**

#### Option A: Using Kaggle (Recommended)
```bash
# Install Kaggle CLI
pip3 install kaggle

# Setup Kaggle credentials
mkdir ~/.kaggle
# Upload your kaggle.json file to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download ImageNet-100 for testing
kaggle datasets download -d ambityga/imagenet100
unzip imagenet100.zip -d ./data/

# Download full ImageNet-1K (if available)
# kaggle competitions download -c imagenet-object-localization-challenge
```

#### Option B: Direct Download
```bash
# If you have ImageNet access, download directly
wget https://your-imagenet-source.com/ILSVRC2012_img_train.tar
wget https://your-imagenet-source.com/ILSVRC2012_img_val.tar

# Extract and organize
mkdir -p ./imagenet_data/{train,val}
# ... extraction and organization scripts
```

### 7. **Training Commands on EC2**

#### Monitor Resources
```bash
# Terminal 1: Monitor GPUs
watch -n 1 nvidia-smi

# Terminal 2: Monitor system resources
htop

# Terminal 3: Run training
```

#### Start Training
```bash
# Set up tmux for persistent sessions
tmux new-session -d -s training

# Find optimal learning rate
python3 train.py --dataset imagenet100 --data-dir ./data --find-lr --batch-size 128 --distributed

# Train ImageNet-100
python3 train.py \
    --dataset imagenet100 \
    --data-dir ./data \
    --epochs 90 \
    --batch-size 128 \
    --max-lr 0.15 \
    --distributed \
    --workers 16

# Scale to ImageNet-1K
python3 train.py \
    --dataset imagenet \
    --data-dir ./imagenet_data \
    --epochs 120 \
    --batch-size 128 \
    --max-lr 0.1 \
    --distributed \
    --workers 16
```

#### Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir ./runs --host 0.0.0.0 --port 6006

# Access via browser: http://ec2-public-ip:6006
# (Make sure port 6006 is open in security group)
```

### 8. **Cost Optimization**

#### Spot Instances
```bash
# Use Spot Instances for 60-90% cost savings
# Request g4dn.8xlarge spot instance
# Set max price slightly above current spot price
```

#### Auto-Shutdown
```bash
# Create auto-shutdown script
echo '#!/bin/bash
if ! pgrep -f "python.*train.py" > /dev/null; then
    echo "Training finished, shutting down in 10 minutes"
    sudo shutdown -h +10
fi' > check_training.sh

# Add to crontab to check every 30 minutes
crontab -e
# Add: */30 * * * * /home/ubuntu/check_training.sh
```

#### Data Transfer
```bash
# Upload results to S3 before shutdown
aws s3 cp ./checkpoints s3://your-bucket/checkpoints/ --recursive
aws s3 cp ./runs s3://your-bucket/runs/ --recursive
```

## 📈 **Expected Results**

| Hardware | Batch Size | Training Time | Expected Accuracy |
|----------|------------|---------------|------------------|
| **2x T4 (g4dn.8xlarge)** | 128×2 | ~6-8 hours | **79.0-79.5%** |
| **Single T4** | 64 | ~12-16 hours | **78.5-79.0%** |
| **V100/A100** | 256×2 | ~4-6 hours | **79.5-80.0%** |

## 🐛 **Troubleshooting**

### Common Issues

#### Out of Memory
```bash
# Reduce batch size
--batch-size 64

# Enable gradient accumulation in code
# accumulation_steps = 2
```

#### Slow Data Loading
```bash
# Increase workers
--workers 16

# Use faster storage (NVMe SSD)
# Mount additional EBS volumes if needed
```

#### Network Issues
```bash
# Check NCCL for distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If InfiniBand issues
```

#### Low Accuracy
```bash
# Verify learning rate with LR finder
--find-lr

# Check data augmentation isn't too aggressive
# Verify normalization statistics match dataset
```

## 📝 **Monitoring**

### TensorBoard Metrics
- Training/Validation Loss
- Top-1 Accuracy
- Learning Rate Schedule
- GPU Utilization

### System Monitoring
```bash
# GPU utilization
nvidia-smi dmon

# Disk I/O
iostat -x 1

# Network traffic
iftop
```

## 🎯 **Performance Tips**

1. **Use SSD storage** for faster data loading
2. **Enable distributed training** on multi-GPU setups
3. **Monitor GPU utilization** - should be >90%
4. **Use appropriate batch size** - larger is generally better
5. **Run learning rate finder** for optimal convergence
6. **Save checkpoints frequently** in case of interruption

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📚 **References**

- [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [RandAugment: Practical automated data augmentation](https://arxiv.org/abs/1909.13719)
- [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
