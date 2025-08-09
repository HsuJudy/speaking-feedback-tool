"""
Demo: Complete Versioning Workflow
Learn how expected hashes are generated and how versioning is tracked
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import our version control system
from data.version_control import DataVersionControl


def demonstrate_versioning_workflow():
    """Demonstrate the complete versioning workflow"""
    print("🔄 COMPLETE VERSIONING WORKFLOW")
    print("=" * 50)
    
    # Initialize version control
    dvc = DataVersionControl("demo_data")
    
    # Step 1: Create a dataset
    print("1. Creating initial dataset...")
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    dataset_path = "demo_data/initial_dataset.csv"
    os.makedirs("demo_data", exist_ok=True)
    data.to_csv(dataset_path, index=False)
    
    # Step 2: Version the dataset (this generates the expected hash)
    print("\n2. Versioning the dataset...")
    version_hash = dvc.version_dataset(
        file_path=dataset_path,
        dataset_name="demo_emotion_dataset",
        description="Initial version of emotion classification dataset",
        tags=["emotion", "audio", "demo"]
    )
    
    print(f"   ✅ Dataset versioned with hash: {version_hash[:16]}...")
    print(f"   📁 This hash is now the 'expected hash' for this dataset")
    
    # Step 3: Show where the hash is stored
    print("\n3. Where is the hash stored?")
    version_file = "demo_data/data_versions.json"
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            versions = json.load(f)
        
        print(f"   📄 Version file: {version_file}")
        print(f"   📊 Number of versioned datasets: {len(versions)}")
        
        # Show the stored version entry
        if version_hash in versions:
            entry = versions[version_hash]
            print(f"   📋 Stored metadata:")
            print(f"      - Dataset name: {entry['dataset_name']}")
            print(f"      - Description: {entry['description']}")
            print(f"      - File size: {entry['file_size']} bytes")
            print(f"      - Created: {entry['created_at']}")
            print(f"      - Tags: {entry['tags']}")
    
    # Step 4: Demonstrate integrity verification
    print("\n4. Testing integrity verification...")
    
    # Test with the same file (should pass)
    is_valid = dvc.verify_dataset_integrity(dataset_path, version_hash)
    print(f"   ✅ Same file verification: {'PASS' if is_valid else 'FAIL'}")
    
    # Test with a modified file (should fail)
    print("\n5. Creating modified dataset...")
    modified_data = data.copy()
    modified_data.iloc[0, 0] = 999  # Modify one value
    modified_path = "demo_data/modified_dataset.csv"
    modified_data.to_csv(modified_path, index=False)
    
    is_valid_modified = dvc.verify_dataset_integrity(modified_path, version_hash)
    print(f"   ❌ Modified file verification: {'PASS' if is_valid_modified else 'FAIL'}")
    
    # Step 5: Show how to get expected hashes
    print("\n6. How to get expected hashes:")
    print("   Method 1: From version control system")
    all_datasets = dvc.list_datasets()
    for dataset in all_datasets:
        print(f"      - {dataset['dataset_name']}: {dataset['content_hash'][:16]}...")
    
    print("\n   Method 2: From specific dataset info")
    dataset_info = dvc.get_dataset_info(version_hash)
    if dataset_info:
        print(f"      - Expected hash: {dataset_info['content_hash']}")
        print(f"      - Dataset: {dataset_info['dataset_name']}")
    
    # Step 6: Demonstrate data splits with versioning
    print("\n7. Creating versioned data splits...")
    splits = dvc.create_data_split(
        dataset_path=dataset_path,
        split_name="emotion_classification",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"   📊 Created splits:")
    for split_name, split_path in splits.items():
        if split_name != "metadata":
            print(f"      - {split_name}: {split_path}")
    
    # Show that splits are also versioned
    print("\n   🔍 All versioned datasets (including splits):")
    all_datasets = dvc.list_datasets()
    for dataset in all_datasets:
        print(f"      - {dataset['dataset_name']}: {dataset['content_hash'][:16]}...")
    
    # Clean up
    print("\n8. Cleaning up...")
    for file in [dataset_path, modified_path]:
        if os.path.exists(file):
            os.remove(file)
    
    print("✅ Versioning workflow demonstration completed!")


def demonstrate_real_world_usage():
    """Show how this works in real MLOps scenarios"""
    print("\n🎯 REAL-WORLD MLOPS USAGE")
    print("=" * 40)
    
    scenarios = [
        {
            "name": "New Dataset Creation",
            "steps": [
                "1. Create/obtain dataset file",
                "2. Call dvc.version_dataset()",
                "3. Store returned hash as 'expected_hash'",
                "4. Use hash in experiments"
            ]
        },
        {
            "name": "Experiment Reproducibility",
            "steps": [
                "1. Load expected hash from experiment metadata",
                "2. Call dvc.verify_dataset_integrity()",
                "3. Only proceed if verification passes",
                "4. Run experiment with verified dataset"
            ]
        },
        {
            "name": "Team Collaboration",
            "steps": [
                "1. Share dataset file with team",
                "2. Share expected hash with team",
                "3. Each team member verifies integrity",
                "4. All use identical dataset version"
            ]
        },
        {
            "name": "Model Deployment",
            "steps": [
                "1. Store dataset hash with model metadata",
                "2. Verify dataset integrity before deployment",
                "3. Deploy model with verified data",
                "4. Track data lineage for audit"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        for step in scenario['steps']:
            print(f"   {step}")


def show_version_file_structure():
    """Show what the version tracking file looks like"""
    print("\n📄 VERSION FILE STRUCTURE")
    print("=" * 30)
    
    # Create a sample version file
    sample_versions = {
        "abc123def456": {
            "dataset_name": "emotion_dataset_v1",
            "file_path": "data/emotion_dataset.csv",
            "content_hash": "abc123def456",
            "description": "Initial emotion classification dataset",
            "tags": ["emotion", "audio", "classification"],
            "created_at": "2024-01-15T10:30:00",
            "file_size": 1024,
            "file_modified": "2024-01-15T10:30:00"
        },
        "def456ghi789": {
            "dataset_name": "emotion_classification_train",
            "file_path": "data/splits/emotion_classification/train.csv",
            "content_hash": "def456ghi789",
            "description": "Training split for emotion classification",
            "tags": ["train", "split", "emotion"],
            "created_at": "2024-01-15T10:35:00",
            "file_size": 716,
            "file_modified": "2024-01-15T10:35:00"
        }
    }
    
    print("data_versions.json:")
    print(json.dumps(sample_versions, indent=2))
    
    print("\n🔑 Key Points:")
    print("   • Hash is the KEY in the dictionary")
    print("   • Each hash maps to complete metadata")
    print("   • File persists across sessions")
    print("   • Enables lookup by hash")


if __name__ == "__main__":
    demonstrate_versioning_workflow()
    demonstrate_real_world_usage()
    show_version_file_structure() 