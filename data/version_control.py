"""
Data Version Control System
Learn MLOps concepts: data versioning, reproducibility, content hashing
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataVersionControl:
    """
    Data versioning system for MLOps learning
    
    Concepts covered:
    - Content hashing (SHA-256)
    - Data lineage tracking
    - Reproducible data splits
    - Metadata management
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Version tracking file
        self.version_file = self.data_dir / "data_versions.json"
        self.versions = self._load_versions()
        
        logger.info(f"DataVersionControl initialized in {self.data_dir}")
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load existing version tracking"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading versions: {e}")
                return {}
        return {}
    
    def _save_versions(self):
        """Save version tracking"""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions: {e}")
    
    def calculate_content_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 content hash of a file
        
        This is a key MLOps concept: content-based versioning
        - Same content = same hash (reproducible)
        - Different content = different hash
        - Enables data lineage tracking
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def version_dataset(self, 
                       file_path: str, 
                       dataset_name: str,
                       description: str = "",
                       tags: List[str] = None) -> str:
        """
        Version a dataset with content hash
        
        Args:
            file_path: Path to dataset file
            dataset_name: Name for the dataset
            description: Human-readable description
            tags: List of tags for organization
            
        Returns:
            str: Version hash
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Calculate content hash
        content_hash = self.calculate_content_hash(file_path)
        
        if not content_hash:
            raise ValueError("Failed to calculate content hash")
        
        # Create version entry
        version_entry = {
            "dataset_name": dataset_name,
            "file_path": file_path,
            "content_hash": content_hash,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path),
            "file_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }
        
        # Store version
        self.versions[content_hash] = version_entry
        self._save_versions()
        
        logger.info(f"Versioned dataset '{dataset_name}' with hash: {content_hash[:8]}...")
        return content_hash
    
    def get_dataset_info(self, version_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset version"""
        return self.versions.get(version_hash)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all versioned datasets"""
        return list(self.versions.values())
    
    def verify_dataset_integrity(self, file_path: str, expected_hash: str) -> bool:
        """
        Verify dataset integrity using content hash
        
        This is crucial for MLOps: ensuring data hasn't been corrupted
        """
        actual_hash = self.calculate_content_hash(file_path)
        return actual_hash == expected_hash
    
    def create_data_split(self, 
                         dataset_path: str,
                         split_name: str,
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.1,
                         test_ratio: float = 0.1,
                         random_seed: int = 42) -> Dict[str, str]:
        """
        Create reproducible data splits
        
        MLOps concept: Reproducible data splits ensure:
        - Same data always produces same splits
        - Experiments can be reproduced
        - No data leakage between splits
        """
        import pandas as pd
        import numpy as np
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Load data
        if dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            data = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Create reproducible splits
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create splits
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]
        test_data = data.iloc[test_indices]
        
        # Save splits
        split_dir = self.data_dir / "splits" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        test_path = split_dir / "test.csv"
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Version the splits
        train_hash = self.version_dataset(str(train_path), f"{split_name}_train", 
                                        f"Training split for {split_name}")
        val_hash = self.version_dataset(str(val_path), f"{split_name}_val", 
                                      f"Validation split for {split_name}")
        test_hash = self.version_dataset(str(test_path), f"{split_name}_test", 
                                       f"Test split for {split_name}")
        
        # Save split metadata
        split_metadata = {
            "split_name": split_name,
            "original_dataset": dataset_path,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "test_hash": test_hash,
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = split_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        logger.info(f"Created reproducible split '{split_name}' with {len(train_data)} train, "
                   f"{len(val_data)} val, {len(test_data)} test samples")
        
        return {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "metadata": str(metadata_path)
        }


def test_data_versioning():
    """Test the data versioning system"""
    print("📊 TESTING DATA VERSIONING")
    print("=" * 40)
    
    # Initialize version control
    dvc = DataVersionControl("test_data")
    
    # Create a test dataset
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Save test dataset
    test_dataset_path = "test_data/sample_dataset.csv"
    os.makedirs("test_data", exist_ok=True)
    data.to_csv(test_dataset_path, index=False)
    
    # Version the dataset
    print("Versioning dataset...")
    version_hash = dvc.version_dataset(
        file_path=test_dataset_path,
        dataset_name="sample_emotion_dataset",
        description="Sample dataset for emotion classification",
        tags=["emotion", "audio", "test"]
    )
    
    print(f"Dataset versioned with hash: {version_hash}")
    
    # Get dataset info
    info = dvc.get_dataset_info(version_hash)
    print(f"Dataset info: {json.dumps(info, indent=2)}")
    
    # Verify integrity
    is_valid = dvc.verify_dataset_integrity(test_dataset_path, version_hash)
    print(f"Dataset integrity verified: {is_valid}")
    
    # Create data splits
    print("\nCreating reproducible data splits...")
    splits = dvc.create_data_split(
        dataset_path=test_dataset_path,
        split_name="emotion_classification",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"Data splits created: {splits}")
    
    # List all datasets
    print("\nAll versioned datasets:")
    for dataset in dvc.list_datasets():
        print(f"- {dataset['dataset_name']}: {dataset['content_hash'][:8]}...")
    
    print("✅ Data versioning test completed!")


if __name__ == "__main__":
    test_data_versioning() 