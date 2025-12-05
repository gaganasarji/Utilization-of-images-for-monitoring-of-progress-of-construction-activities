"""
Train Custom YOLO Model for Construction Equipment Detection
"""

import os
from ultralytics import YOLO
import yaml

# Dataset configuration
DATASET_CONFIG = {
    'path': './construction_dataset',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 10,  # number of classes
    'names': [
        'excavator', 'bulldozer', 'crane', 'concrete_mixer',
        'dump_truck', 'worker', 'scaffolding', 'materials',
        'safety_helmet', 'safety_vest'
    ]
}

def create_dataset_yaml():
    """Create dataset configuration file"""
    with open('construction.yaml', 'w') as f:
        yaml.dump(DATASET_CONFIG, f)
    print("Dataset configuration created: construction.yaml")

def train_model(epochs=100, img_size=640, batch_size=16):
    """Train YOLO model on construction dataset"""
    
    # Create dataset config
    create_dataset_yaml()
    
    # Load pretrained YOLO model
    model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano
    
    print("Starting training...")
    results = model.train(
        data='construction.yaml',
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='construction_detector',
        project='runs/train',
        patience=50,
        save=True,
        device=0,  # Use GPU 0, or 'cpu' for CPU training
        workers=8,
        optimizer='Adam',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    print("Training completed!")
    print(f"Best model saved to: runs/train/construction_detector/weights/best.pt")
    
    return results

def validate_model(model_path='runs/train/construction_detector/weights/best.pt'):
    """Validate trained model"""
    model = YOLO(model_path)
    
    print("Validating model...")
    metrics = model.val(data='construction.yaml')
    
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return metrics

def export_model(model_path='runs/train/construction_detector/weights/best.pt'):
    """Export model to different formats"""
    model = YOLO(model_path)
    
    # Export to ONNX
    model.export(format='onnx')
    print("Model exported to ONNX format")
    
    # Export to TensorRT (if available)
    try:
        model.export(format='engine')
        print("Model exported to TensorRT format")
    except Exception as e:
        print(f"TensorRT export failed: {e}")

def test_inference(model_path='runs/train/construction_detector/weights/best.pt',
                   image_path='test_image.jpg'):
    """Test model inference on a sample image"""
    model = YOLO(model_path)
    
    results = model(image_path)
    
    # Display results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects")
        
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            print(f"  - {class_name}: {confidence:.2f}")
    
    # Save annotated image
    results[0].save('test_result.jpg')
    print("Annotated image saved to test_result.jpg")

if _name_ == '_main_':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train construction equipment detector')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--validate', action='store_true', help='Validate model')
    parser.add_argument('--export', action='store_true', help='Export model')
    parser.add_argument('--test', type=str, help='Test inference on image')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--model', type=str, 
                       default='runs/train/construction_detector/weights/best.pt',
                       help='Model path')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(epochs=args.epochs, img_size=args.img_size, batch_size=args.batch)
    
    if args.validate:
        validate_model(args.model)
    
    if args.export:
        export_model(args.model)
    
    if args.test:
        test_inference(args.model, args.test)
    
    if not any([args.train, args.validate, args.export, args.test]):
        print("Usage: python train_model.py --train --epochs 100")
        print("       python train_model.py --validate --model path/to/model.pt")
        print("       python train_model.py --test path/to/image.jpg")