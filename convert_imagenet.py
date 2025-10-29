# convert_imagenet.py
import os
from datasets import load_from_disk
from PIL import Image
import argparse

def convert_arrow_to_imagenet_structure(data_path, output_path):
    """Convert HuggingFace Arrow format to ImageNet folder structure"""
    print("Loading dataset from disk...")
    dataset = load_from_disk(data_path)
    
    # Create output directories
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get unique class names
    if 'train' in dataset:
        train_data = dataset['train']
        class_names = train_data.features['label'].names
        
        # Create class directories
        for class_name in class_names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Convert training data
        print("Converting training data...")
        for i, sample in enumerate(train_data):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(train_data)} training samples")
            
            image = sample['image']
            label = sample['label']
            class_name = class_names[label]
            
            # Save image
            image_path = os.path.join(train_dir, class_name, f"{i:06d}.JPEG")
            image.save(image_path, 'JPEG')
    
    # Convert validation data
    if 'validation' in dataset:
        val_data = dataset['validation']
        print("Converting validation data...")
        for i, sample in enumerate(val_data):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(val_data)} validation samples")
            
            image = sample['image']
            label = sample['label']
            class_name = class_names[label]
            
            # Save image
            image_path = os.path.join(val_dir, class_name, f"{i:06d}.JPEG")
            image.save(image_path, 'JPEG')
    
    print(f"Conversion complete! Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Arrow dataset")
    parser.add_argument("--output", required=True, help="Output path for ImageNet structure")
    args = parser.parse_args()
    
    convert_arrow_to_imagenet_structure(args.input, args.output)