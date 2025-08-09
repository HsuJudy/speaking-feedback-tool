"""
Demo: Multi-Dataset Management with DVC
Learn how one DVC instance manages multiple datasets
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import our version control system
from data.version_control import DataVersionControl


def demonstrate_multi_dataset_management():
    """Demonstrate how one DVC instance manages multiple datasets"""
    print("📊 MULTI-DATASET MANAGEMENT WITH DVC")
    print("=" * 50)
    
    # Initialize ONE DVC instance for ALL datasets
    dvc = DataVersionControl("multi_demo_data")
    
    # Create multiple datasets
    print("1. Creating multiple datasets...")
    
    # Dataset 1: Emotion data
    np.random.seed(42)
    emotion_data = pd.DataFrame({
        'pitch_mean': np.random.normal(200, 50, 100),
        'volume_mean': np.random.uniform(0.1, 0.9, 100),
        'emotion': np.random.choice(['calm', 'anxious', 'frustrated', 'energetic', 'burned_out'], 100)
    })
    
    # Dataset 2: Audio features
    audio_features = pd.DataFrame({
        'spectral_centroid': np.random.normal(2000, 500, 80),
        'zero_crossing_rate': np.random.uniform(0.05, 0.15, 80),
        'mfcc_1': np.random.normal(0, 1, 80),
        'mfcc_2': np.random.normal(0, 1, 80)
    })
    
    # Dataset 3: Validation data
    validation_data = pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'target': np.random.randint(0, 2, 50)
    })
    
    # Save datasets
    os.makedirs("multi_demo_data", exist_ok=True)
    emotion_data.to_csv("multi_demo_data/emotion_dataset.csv", index=False)
    audio_features.to_csv("multi_demo_data/audio_features.csv", index=False)
    validation_data.to_csv("multi_demo_data/validation_data.csv", index=False)
    
    # Version ALL datasets with the SAME DVC instance
    print("\n2. Versioning all datasets with one DVC instance...")
    
    hash1 = dvc.version_dataset(
        file_path="multi_demo_data/emotion_dataset.csv",
        dataset_name="emotion_classification_data",
        description="Emotion classification dataset with audio features",
        tags=["emotion", "audio", "classification"]
    )
    
    hash2 = dvc.version_dataset(
        file_path="multi_demo_data/audio_features.csv", 
        dataset_name="audio_feature_dataset",
        description="Raw audio features for analysis",
        tags=["audio", "features", "spectral"]
    )
    
    hash3 = dvc.version_dataset(
        file_path="multi_demo_data/validation_data.csv",
        dataset_name="validation_dataset", 
        description="Validation dataset for model testing",
        tags=["validation", "testing"]
    )
    
    print(f"   ✅ Emotion dataset hash: {hash1[:16]}...")
    print(f"   ✅ Audio features hash: {hash2[:16]}...")
    print(f"   ✅ Validation dataset hash: {hash3[:16]}...")
    
    # Show how all datasets are stored together
    print("\n3. All datasets stored in one version file:")
    version_file = "multi_demo_data/data_versions.json"
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            all_versions = json.load(f)
        
        print(f"   📄 Version file: {version_file}")
        print(f"   📊 Total datasets managed: {len(all_versions)}")
        
        for hash_value, metadata in all_versions.items():
            print(f"   📋 {metadata['dataset_name']}: {hash_value[:16]}...")
    
    # Demonstrate querying specific datasets
    print("\n4. Querying specific datasets:")
    
    # Get info for specific dataset
    emotion_info = dvc.get_dataset_info(hash1)
    if emotion_info:
        print(f"   🎯 Emotion dataset info:")
        print(f"      - Name: {emotion_info['dataset_name']}")
        print(f"      - Description: {emotion_info['description']}")
        print(f"      - Tags: {emotion_info['tags']}")
        print(f"      - File size: {emotion_info['file_size']} bytes")
    
    # List all datasets
    print("\n5. Listing all managed datasets:")
    all_datasets = dvc.list_datasets()
    for i, dataset in enumerate(all_datasets, 1):
        print(f"   {i}. {dataset['dataset_name']}")
        print(f"      - Hash: {dataset['content_hash'][:16]}...")
        print(f"      - Tags: {dataset['tags']}")
        print(f"      - Created: {dataset['created_at'][:10]}")
    
    # Demonstrate data splits for multiple datasets
    print("\n6. Creating data splits for multiple datasets:")
    
    # Split emotion dataset
    emotion_splits = dvc.create_data_split(
        dataset_path="multi_demo_data/emotion_dataset.csv",
        split_name="emotion_classification",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Split validation dataset
    validation_splits = dvc.create_data_split(
        dataset_path="multi_demo_data/validation_data.csv", 
        split_name="validation_testing",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    print(f"   📊 Created splits for emotion dataset")
    print(f"   📊 Created splits for validation dataset")
    
    # Show total number of versioned items (including splits)
    print("\n7. Total versioned items (including splits):")
    final_datasets = dvc.list_datasets()
    print(f"   📈 Total items managed: {len(final_datasets)}")
    
    # Group by dataset type
    dataset_types = {}
    for dataset in final_datasets:
        name = dataset['dataset_name']
        if 'emotion' in name:
            dataset_types.setdefault('emotion', []).append(name)
        elif 'audio' in name:
            dataset_types.setdefault('audio', []).append(name)
        elif 'validation' in name:
            dataset_types.setdefault('validation', []).append(name)
    
    for dataset_type, names in dataset_types.items():
        print(f"   📁 {dataset_type.title()} datasets: {len(names)}")
        for name in names:
            print(f"      - {name}")
    
    # Demonstrate integrity verification for multiple datasets
    print("\n8. Verifying integrity for all datasets:")
    
    datasets_to_verify = [
        ("multi_demo_data/emotion_dataset.csv", hash1, "Emotion dataset"),
        ("multi_demo_data/audio_features.csv", hash2, "Audio features"),
        ("multi_demo_data/validation_data.csv", hash3, "Validation data")
    ]
    
    for file_path, expected_hash, dataset_name in datasets_to_verify:
        is_valid = dvc.verify_dataset_integrity(file_path, expected_hash)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"   {status} {dataset_name}")
    
    # Clean up
    print("\n9. Cleaning up...")
    for file in ["multi_demo_data/emotion_dataset.csv", 
                 "multi_demo_data/audio_features.csv",
                 "multi_demo_data/validation_data.csv"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("✅ Multi-dataset management demonstration completed!")


def demonstrate_project_organization():
    """Show how to organize datasets in a real project"""
    print("\n🏗️ PROJECT ORGANIZATION WITH DVC")
    print("=" * 40)
    
    print("📁 Typical project structure:")
    print("   project/")
    print("   ├── data/")
    print("   │   ├── data_versions.json          # ONE version file for ALL datasets")
    print("   │   ├── raw/")
    print("   │   │   ├── emotion_dataset.csv")
    print("   │   │   ├── audio_features.csv")
    print("   │   │   └── validation_data.csv")
    print("   │   ├── processed/")
    print("   │   │   ├── emotion_processed.csv")
    print("   │   │   └── audio_processed.csv")
    print("   │   └── splits/")
    print("   │       ├── emotion_classification/")
    print("   │       │   ├── train.csv")
    print("   │       │   ├── val.csv")
    print("   │       │   └── test.csv")
    print("   │       └── validation_testing/")
    print("   │           ├── train.csv")
    print("   │           ├── val.csv")
    print("   │           └── test.csv")
    print("   ├── models/")
    print("   ├── notebooks/")
    print("   └── scripts/")
    
    print("\n🔑 Key Benefits:")
    print("   • ONE DVC instance manages ALL datasets")
    print("   • Centralized version tracking")
    print("   • Easy to see all dataset versions")
    print("   • Consistent data integrity checks")
    print("   • Simplified project management")


def demonstrate_team_collaboration():
    """Show how DVC works in team environments"""
    print("\n👥 TEAM COLLABORATION WITH DVC")
    print("=" * 40)
    
    scenarios = [
        {
            "situation": "New team member joins",
            "steps": [
                "1. Share data_versions.json file",
                "2. Share all dataset files",
                "3. New member loads same DVC instance",
                "4. Verify all datasets match hashes",
                "5. Start working with verified data"
            ]
        },
        {
            "situation": "Dataset updates",
            "steps": [
                "1. Update dataset file",
                "2. Re-version with DVC",
                "3. Share new hash with team",
                "4. Team updates their expected hashes",
                "5. All use new dataset version"
            ]
        },
        {
            "situation": "Experiment reproduction",
            "steps": [
                "1. Load experiment metadata (includes dataset hashes)",
                "2. Use DVC to verify all datasets",
                "3. Only proceed if all verifications pass",
                "4. Reproduce exact experiment conditions"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['situation']}:")
        for step in scenario['steps']:
            print(f"   {step}")


if __name__ == "__main__":
    demonstrate_multi_dataset_management()
    demonstrate_project_organization()
    demonstrate_team_collaboration() 