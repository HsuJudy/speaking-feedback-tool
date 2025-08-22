"""
Advanced Data Split Manager for MLOps
Implements comprehensive data splitting strategies for NVIDIA MLOps interviews
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedShuffleSplit, KFold, 
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import LabelEncoder
import joblib

logger = logging.getLogger(__name__)


class AdvancedSplitManager:
    """
    Advanced data split manager implementing all NVIDIA MLOps interview requirements
    
    Features:
    - Reproducible splits with integrity verification
    - Stratified splitting for imbalanced datasets
    - Large dataset handling with streaming
    - Cross-validation integration
    - Performance impact evaluation
    - Team collaboration workflows
    """
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Split registry
        self.split_registry_file = self.data_dir / "split_registry.json"
        self.split_registry = self._load_split_registry()
        
        logger.info(f"AdvancedSplitManager initialized in {self.data_dir}")
    
    def _load_split_registry(self) -> Dict[str, Any]:
        """Load split registry for versioning"""
        if self.split_registry_file.exists():
            try:
                with open(self.split_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading split registry: {e}")
                return {}
        return {}
    
    def _save_split_registry(self):
        """Save split registry"""
        try:
            with open(self.split_registry_file, 'w') as f:
                json.dump(self.split_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving split registry: {e}")
    
    def calculate_content_hash(self, file_path: str) -> str:
        """Calculate SHA-256 content hash of a file"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def create_reproducible_split(self, 
                                 dataset_path: str,
                                 split_name: str,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_seed: int = 42,
                                 stratify_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Create reproducible data splits with comprehensive metadata
        
        This answers: "How do you ensure your data splits are reproducible?"
        """
        logger.info(f"Creating reproducible split '{split_name}' with seed {random_seed}")
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Load data
        if dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            data = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Create splits
        if stratify_column and stratify_column in data.columns:
            # Stratified split for imbalanced datasets
            train_data, temp_data = train_test_split(
                data,
                test_size=(val_ratio + test_ratio),
                stratify=data[stratify_column],
                random_state=random_seed
            )
            
            # Split remaining data
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                stratify=temp_data[stratify_column],
                random_state=random_seed
            )
        else:
            # Random split
            train_data, temp_data = train_test_split(
                data,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed
            )
            
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_seed
            )
        
        # Save splits
        split_dir = self.data_dir / "splits" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        test_path = split_dir / "test.csv"
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Calculate hashes for integrity verification
        train_hash = self.calculate_content_hash(str(train_path))
        val_hash = self.calculate_content_hash(str(val_path))
        test_hash = self.calculate_content_hash(str(test_path))
        
        # Create comprehensive split metadata
        split_metadata = {
            "split_name": split_name,
            "original_dataset": dataset_path,
            "original_dataset_hash": self.calculate_content_hash(dataset_path),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "stratify_column": stratify_column,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "test_hash": test_hash,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "created_at": datetime.now().isoformat(),
            "split_parameters": {
                "stratify": stratify_column is not None,
                "shuffle": True,
                "n_splits": 1
            },
            "data_statistics": {
                "total_samples": len(data),
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
                "features": list(data.columns)
            }
        }
        
        # Add class distribution if stratified
        if stratify_column:
            split_metadata["class_distribution"] = {
                "train": train_data[stratify_column].value_counts().to_dict(),
                "val": val_data[stratify_column].value_counts().to_dict(),
                "test": test_data[stratify_column].value_counts().to_dict()
            }
        
        # Save metadata
        metadata_path = split_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        # Register split
        split_id = f"{split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.split_registry[split_id] = split_metadata
        self._save_split_registry()
        
        logger.info(f"Created reproducible split '{split_name}' with {len(train_data)} train, "
                   f"{len(val_data)} val, {len(test_data)} test samples")
        
        return {
            "split_id": split_id,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "metadata_path": str(metadata_path),
            "metadata": split_metadata
        }
    
    def verify_split_integrity(self, split_metadata_path: str) -> bool:
        """
        Verify split integrity using stored hashes
        
        This answers: "How do you verify that your data splits haven't been corrupted?"
        """
        try:
            with open(split_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify each split file
            train_valid = self.calculate_content_hash(metadata["train_path"]) == metadata["train_hash"]
            val_valid = self.calculate_content_hash(metadata["val_path"]) == metadata["val_hash"]
            test_valid = self.calculate_content_hash(metadata["test_path"]) == metadata["test_hash"]
            
            # Verify original dataset
            original_valid = self.calculate_content_hash(metadata["original_dataset"]) == metadata["original_dataset_hash"]
            
            all_valid = train_valid and val_valid and test_valid and original_valid
            
            if not all_valid:
                logger.error(f"Split integrity verification failed for {split_metadata_path}")
                logger.error(f"Train valid: {train_valid}, Val valid: {val_valid}, "
                           f"Test valid: {test_valid}, Original valid: {original_valid}")
            
            return all_valid
            
        except Exception as e:
            logger.error(f"Error verifying split integrity: {e}")
            return False
    
    def create_large_dataset_split(self, 
                                  dataset_path: str,
                                  split_name: str,
                                  chunk_size: int = 10000,
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15,
                                  random_seed: int = 42) -> Dict[str, Any]:
        """
        Handle large datasets that don't fit in memory
        
        This answers: "How do you handle data splits for very large datasets?"
        """
        logger.info(f"Creating large dataset split '{split_name}' with chunk size {chunk_size}")
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Calculate total rows without loading entire dataset
        total_rows = sum(1 for _ in pd.read_csv(dataset_path, chunksize=chunk_size))
        
        # Create split indices
        indices = np.random.permutation(total_rows)
        train_end = int(total_rows * train_ratio)
        val_end = train_end + int(total_rows * val_ratio)
        
        train_indices = set(indices[:train_end])
        val_indices = set(indices[train_end:val_end])
        test_indices = set(indices[val_end:])
        
        # Create output directories
        split_dir = self.data_dir / "splits" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        test_path = split_dir / "test.csv"
        
        # Process dataset in chunks
        train_chunks = []
        val_chunks = []
        test_chunks = []
        chunk_hashes = []
        
        for chunk_idx, chunk in enumerate(pd.read_csv(dataset_path, chunksize=chunk_size)):
            # Calculate chunk hash
            chunk_hash = hashlib.sha256(chunk.to_csv(index=False).encode()).hexdigest()
            chunk_hashes.append(chunk_hash)
            
            # Determine which split this chunk belongs to
            chunk_start_idx = chunk_idx * chunk_size
            chunk_end_idx = chunk_start_idx + len(chunk)
            
            # Create mask for this chunk
            chunk_indices = set(range(chunk_start_idx, chunk_end_idx))
            
            train_mask = chunk_indices & train_indices
            val_mask = chunk_indices & val_indices
            test_mask = chunk_indices & test_indices
            
            # Split chunk
            if train_mask:
                train_chunk = chunk.iloc[list(train_mask - set(range(chunk_start_idx)))]
                train_chunks.append(train_chunk)
            
            if val_mask:
                val_chunk = chunk.iloc[list(val_mask - set(range(chunk_start_idx)))]
                val_chunks.append(val_chunk)
            
            if test_mask:
                test_chunk = chunk.iloc[list(test_mask - set(range(chunk_start_idx)))]
                test_chunks.append(test_chunk)
        
        # Combine chunks and save
        train_data = pd.concat(train_chunks, ignore_index=True) if train_chunks else pd.DataFrame()
        val_data = pd.concat(val_chunks, ignore_index=True) if val_chunks else pd.DataFrame()
        test_data = pd.concat(test_chunks, ignore_index=True) if test_chunks else pd.DataFrame()
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Calculate hashes
        train_hash = self.calculate_content_hash(str(train_path))
        val_hash = self.calculate_content_hash(str(val_path))
        test_hash = self.calculate_content_hash(str(test_path))
        
        # Create metadata
        split_metadata = {
            "split_name": split_name,
            "original_dataset": dataset_path,
            "chunk_size": chunk_size,
            "total_rows": total_rows,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "test_hash": test_hash,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "chunk_hashes": chunk_hashes,
            "created_at": datetime.now().isoformat(),
            "split_parameters": {
                "large_dataset": True,
                "chunk_size": chunk_size,
                "streaming": True
            },
            "data_statistics": {
                "total_samples": total_rows,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data)
            }
        }
        
        # Save metadata
        metadata_path = split_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        logger.info(f"Created large dataset split with {len(train_data)} train, "
                   f"{len(val_data)} val, {len(test_data)} test samples")
        
        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "metadata_path": str(metadata_path),
            "metadata": split_metadata
        }
    
    def create_cv_splits(self, 
                        dataset_path: str,
                        split_name: str,
                        n_splits: int = 5,
                        random_seed: int = 42,
                        stratify_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Create cross-validation splits with comprehensive metadata
        
        This answers: "How do you integrate cross-validation with your split metadata?"
        """
        logger.info(f"Creating CV splits '{split_name}' with {n_splits} folds")
        
        # Load data
        if dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Create CV splits
        if stratify_column and stratify_column in data.columns:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            splits = cv.split(data, data[stratify_column])
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            splits = cv.split(data)
        
        # Create output directory
        split_dir = self.data_dir / "splits" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        cv_metadata = {
            "split_name": split_name,
            "original_dataset": dataset_path,
            "original_dataset_hash": self.calculate_content_hash(dataset_path),
            "n_splits": n_splits,
            "random_seed": random_seed,
            "stratify_column": stratify_column,
            "created_at": datetime.now().isoformat(),
            "fold_hashes": {},
            "fold_paths": {},
            "fold_indices": {},
            "fold_statistics": {}
        }
        
        # Process each fold
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            
            # Save fold data
            train_path = split_dir / f"fold_{fold}_train.csv"
            val_path = split_dir / f"fold_{fold}_val.csv"
            
            train_fold.to_csv(train_path, index=False)
            val_fold.to_csv(val_path, index=False)
            
            # Calculate hashes
            train_hash = self.calculate_content_hash(str(train_path))
            val_hash = self.calculate_content_hash(str(val_path))
            
            # Store metadata
            cv_metadata["fold_hashes"][fold] = {
                "train_hash": train_hash,
                "val_hash": val_hash
            }
            cv_metadata["fold_paths"][fold] = {
                "train_path": str(train_path),
                "val_path": str(val_path)
            }
            cv_metadata["fold_indices"][fold] = {
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            }
            cv_metadata["fold_statistics"][fold] = {
                "train_samples": len(train_fold),
                "val_samples": len(val_fold)
            }
            
            # Add class distribution if stratified
            if stratify_column:
                cv_metadata["fold_statistics"][fold]["class_distribution"] = {
                    "train": train_fold[stratify_column].value_counts().to_dict(),
                    "val": val_fold[stratify_column].value_counts().to_dict()
                }
        
        # Save CV metadata
        metadata_path = split_dir / "cv_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(cv_metadata, f, indent=2)
        
        logger.info(f"Created CV splits with {n_splits} folds")
        
        return {
            "metadata_path": str(metadata_path),
            "metadata": cv_metadata,
            "split_dir": str(split_dir)
        }
    
    def evaluate_split_strategies(self, 
                                 dataset_path: str,
                                 target_column: str,
                                 strategies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate different split strategies on model performance
        
        This answers: "How do you measure the impact of different split strategies?"
        """
        if strategies is None:
            strategies = [
                {"name": "random", "stratify": False, "random_seed": 42},
                {"name": "stratified", "stratify": True, "stratify_column": target_column, "random_seed": 42},
                {"name": "stratified_different_seed", "stratify": True, "stratify_column": target_column, "random_seed": 123}
            ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"Evaluating strategy: {strategy['name']}")
            
            # Create splits with this strategy
            split_name = f"strategy_eval_{strategy['name']}"
            
            if strategy.get("stratify"):
                splits = self.create_reproducible_split(
                    dataset_path=dataset_path,
                    split_name=split_name,
                    stratify_column=strategy.get("stratify_column"),
                    random_seed=strategy.get("random_seed", 42)
                )
            else:
                splits = self.create_reproducible_split(
                    dataset_path=dataset_path,
                    split_name=split_name,
                    random_seed=strategy.get("random_seed", 42)
                )
            
            # Train and evaluate model (simplified for demo)
            performance = self._evaluate_split_performance(
                splits["train_path"], 
                splits["val_path"], 
                target_column
            )
            
            results[strategy["name"]] = {
                "splits": splits,
                "performance": performance,
                "strategy_config": strategy
            }
        
        # Save evaluation results
        eval_path = self.data_dir / "split_strategy_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _evaluate_split_performance(self, train_path: str, val_path: str, target_column: str) -> Dict[str, float]:
        """Evaluate model performance on given splits"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Load data
            train_data = pd.read_csv(train_path)
            val_data = pd.read_csv(val_path)
            
            # Prepare features and target
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_val = val_data.drop(columns=[target_column])
            y_val = val_data[target_column]
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            performance = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, average='weighted'),
                "recall": recall_score(y_val, y_pred, average='weighted'),
                "f1_score": f1_score(y_val, y_pred, average='weighted')
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating split performance: {e}")
            return {"error": str(e)}
    
    def share_splits_with_team(self, split_metadata_path: str) -> Dict[str, Any]:
        """
        Share splits with team members for collaboration
        
        This answers: "How do multiple team members ensure they're using the same data splits?"
        """
        logger.info(f"Preparing splits for team sharing: {split_metadata_path}")
        
        with open(split_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify original dataset integrity
        original_valid = self.calculate_content_hash(metadata["original_dataset"]) == metadata["original_dataset_hash"]
        
        if not original_valid:
            raise ValueError("Original dataset integrity verification failed")
        
        # Recreate splits with same parameters
        splits = self.create_reproducible_split(
            dataset_path=metadata["original_dataset"],
            split_name=f"{metadata['split_name']}_team_shared",
            train_ratio=metadata["train_ratio"],
            val_ratio=metadata["val_ratio"],
            test_ratio=metadata["test_ratio"],
            random_seed=metadata["random_seed"],
            stratify_column=metadata.get("stratify_column")
        )
        
        # Verify recreated splits match
        recreated_metadata = splits["metadata"]
        train_match = recreated_metadata["train_hash"] == metadata["train_hash"]
        val_match = recreated_metadata["val_hash"] == metadata["val_hash"]
        test_match = recreated_metadata["test_hash"] == metadata["test_hash"]
        
        if not (train_match and val_match and test_match):
            raise ValueError("Recreated splits do not match original splits")
        
        # Create team sharing package
        sharing_package = {
            "original_dataset_path": metadata["original_dataset"],
            "original_dataset_hash": metadata["original_dataset_hash"],
            "split_metadata": metadata,
            "verification_results": {
                "original_dataset_valid": original_valid,
                "train_split_match": train_match,
                "val_split_match": val_match,
                "test_split_match": test_match,
                "all_verified": True
            },
            "team_instructions": [
                "1. Download the original dataset",
                "2. Verify dataset hash matches original_dataset_hash",
                "3. Use split_metadata to recreate splits",
                "4. Verify recreated splits match stored hashes"
            ],
            "created_at": datetime.now().isoformat()
        }
        
        # Save sharing package
        sharing_path = self.data_dir / "team_sharing" / f"{metadata['split_name']}_sharing_package.json"
        sharing_path.parent.mkdir(exist_ok=True)
        
        with open(sharing_path, 'w') as f:
            json.dump(sharing_package, f, indent=2)
        
        logger.info(f"Team sharing package created: {sharing_path}")
        
        return sharing_package
    
    def list_all_splits(self) -> List[Dict[str, Any]]:
        """List all created splits with their metadata"""
        return list(self.split_registry.values())
    
    def get_split_info(self, split_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific split"""
        return self.split_registry.get(split_id)
    
    def delete_split(self, split_id: str) -> bool:
        """Delete a split and its files"""
        if split_id not in self.split_registry:
            return False
        
        split_info = self.split_registry[split_id]
        
        # Delete split files
        for path_key in ["train_path", "val_path", "test_path"]:
            if path_key in split_info:
                try:
                    os.remove(split_info[path_key])
                except FileNotFoundError:
                    pass
        
        # Delete metadata file
        metadata_path = Path(split_info.get("metadata_path", ""))
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from registry
        del self.split_registry[split_id]
        self._save_split_registry()
        
        logger.info(f"Deleted split: {split_id}")
        return True


def test_advanced_split_manager():
    """Test the advanced split manager with all features"""
    print("🧪 TESTING ADVANCED SPLIT MANAGER")
    print("=" * 50)
    
    # Initialize manager
    manager = AdvancedSplitManager("test_advanced_data")
    
    # Create test dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000),
        'target': np.random.choice(['class_0', 'class_1', 'class_2'], 1000, p=[0.6, 0.3, 0.1])
    })
    
    dataset_path = "test_advanced_data/test_dataset.csv"
    os.makedirs("test_advanced_data", exist_ok=True)
    data.to_csv(dataset_path, index=False)
    
    print("1. Testing reproducible splits...")
    splits = manager.create_reproducible_split(
        dataset_path=dataset_path,
        split_name="test_reproducible",
        stratify_column="target"
    )
    print(f"   ✅ Created reproducible split: {splits['split_id']}")
    
    print("\n2. Testing split integrity verification...")
    is_valid = manager.verify_split_integrity(splits["metadata_path"])
    print(f"   ✅ Split integrity verified: {is_valid}")
    
    print("\n3. Testing large dataset splits...")
    large_splits = manager.create_large_dataset_split(
        dataset_path=dataset_path,
        split_name="test_large_dataset",
        chunk_size=100
    )
    print(f"   ✅ Created large dataset split")
    
    print("\n4. Testing CV splits...")
    cv_splits = manager.create_cv_splits(
        dataset_path=dataset_path,
        split_name="test_cv",
        n_splits=3,
        stratify_column="target"
    )
    print(f"   ✅ Created CV splits with 3 folds")
    
    print("\n5. Testing split strategy evaluation...")
    strategy_results = manager.evaluate_split_strategies(
        dataset_path=dataset_path,
        target_column="target"
    )
    print(f"   ✅ Evaluated {len(strategy_results)} split strategies")
    
    print("\n6. Testing team sharing...")
    sharing_package = manager.share_splits_with_team(splits["metadata_path"])
    print(f"   ✅ Created team sharing package")
    
    print("\n7. Testing split listing...")
    all_splits = manager.list_all_splits()
    print(f"   ✅ Found {len(all_splits)} total splits")
    
    print("\n✅ Advanced split manager test completed!")


if __name__ == "__main__":
    test_advanced_split_manager()
