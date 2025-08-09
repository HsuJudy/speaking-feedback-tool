"""
Demo: Content Hash Validation for MLOps
Learn how content hashing ensures data integrity and reproducibility
"""

import hashlib
import json
import os
from datetime import datetime


def demonstrate_content_hashing():
    """Demonstrate content hashing concepts"""
    print("🔍 CONTENT HASHING DEMONSTRATION")
    print("=" * 50)
    
    # Create test files
    print("1. Creating test files...")
    
    # File 1: Original dataset
    dataset_1 = {
        "features": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
        "metadata": {"version": "1.0", "created": "2024-01-01"}
    }
    
    with open("test_dataset_1.json", "w") as f:
        json.dump(dataset_1, f, indent=2)
    
    # File 2: Identical content (should have same hash)
    dataset_2 = {
        "features": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
        "metadata": {"version": "1.0", "created": "2024-01-01"}
    }
    
    with open("test_dataset_2.json", "w") as f:
        json.dump(dataset_2, f, indent=2)
    
    # File 3: Different content (should have different hash)
    dataset_3 = {
        "features": [1, 2, 3, 4, 6],  # Changed last value
        "target": [0, 1, 0, 1, 0],
        "metadata": {"version": "1.0", "created": "2024-01-01"}
    }
    
    with open("test_dataset_3.json", "w") as f:
        json.dump(dataset_3, f, indent=2)
    
    # Calculate hashes
    print("\n2. Calculating content hashes...")
    
    def calculate_hash(file_path):
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    hash_1 = calculate_hash("test_dataset_1.json")
    hash_2 = calculate_hash("test_dataset_2.json")
    hash_3 = calculate_hash("test_dataset_3.json")
    
    print(f"Dataset 1 hash: {hash_1[:16]}...")
    print(f"Dataset 2 hash: {hash_2[:16]}...")
    print(f"Dataset 3 hash: {hash_3[:16]}...")
    
    # Demonstrate key concepts
    print("\n3. Key MLOps Concepts:")
    print(f"   ✅ Same content = Same hash: {hash_1 == hash_2}")
    print(f"   ✅ Different content = Different hash: {hash_1 != hash_3}")
    print(f"   ✅ Hash length: {len(hash_1)} characters")
    
    # Demonstrate data integrity verification
    print("\n4. Data Integrity Verification:")
    
    def verify_integrity(file_path, expected_hash):
        actual_hash = calculate_hash(file_path)
        is_valid = actual_hash == expected_hash
        print(f"   File: {file_path}")
        print(f"   Expected: {expected_hash[:16]}...")
        print(f"   Actual:   {actual_hash[:16]}...")
        print(f"   Valid:    {'✅' if is_valid else '❌'}")
        return is_valid
    
    print("\n   Testing integrity verification:")
    verify_integrity("test_dataset_1.json", hash_1)  # Should pass
    verify_integrity("test_dataset_2.json", hash_1)  # Should pass (same content)
    verify_integrity("test_dataset_3.json", hash_1)  # Should fail (different content)
    
    # Demonstrate reproducibility
    print("\n5. Reproducibility Test:")
    print("   Running hash calculation multiple times...")
    
    for i in range(3):
        hash_repeat = calculate_hash("test_dataset_1.json")
        print(f"   Attempt {i+1}: {hash_repeat[:16]}...")
    
    print("   ✅ Same file always produces same hash (deterministic)")
    
    # Clean up
    print("\n6. Cleaning up test files...")
    for file in ["test_dataset_1.json", "test_dataset_2.json", "test_dataset_3.json"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("✅ Content hashing demonstration completed!")


def demonstrate_mlops_scenarios():
    """Demonstrate real MLOps scenarios where content hashing is crucial"""
    print("\n🎯 MLOPS SCENARIOS WHERE CONTENT HASHING MATTERS")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Data Corruption Detection",
            "description": "Detect if dataset was corrupted during transfer",
            "example": "File transfer over network, storage failure, etc."
        },
        {
            "name": "Experiment Reproducibility",
            "description": "Ensure you're using the exact same dataset version",
            "example": "Re-running experiments months later"
        },
        {
            "name": "Data Lineage Tracking",
            "description": "Track which dataset version was used for each model",
            "example": "Audit trail for model deployments"
        },
        {
            "name": "Collaboration Safety",
            "description": "Ensure team members use identical datasets",
            "example": "Multiple data scientists working on same project"
        },
        {
            "name": "Model Versioning",
            "description": "Link model versions to specific dataset versions",
            "example": "Rollback to previous model with exact same data"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   📝 {scenario['description']}")
        print(f"   💡 Example: {scenario['example']}")
    
    print("\n🔑 Key Benefits:")
    print("   • Prevents silent data corruption")
    print("   • Enables exact experiment reproduction")
    print("   • Provides audit trail for compliance")
    print("   • Ensures team collaboration consistency")
    print("   • Supports model versioning and rollbacks")


if __name__ == "__main__":
    demonstrate_content_hashing()
    demonstrate_mlops_scenarios() 