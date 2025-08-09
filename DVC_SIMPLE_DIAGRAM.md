# DVC Simple Workflow Diagram

## 🔄 Core DVC Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DVC WORKFLOW                                │
└─────────────────────────────────────────────────────────────────┘

📁 Raw Dataset
    │
    ▼
┌─────────────────┐
│   DVC Version   │  ← version_dataset()
│   Control       │
└─────────────────┘
    │
    ▼
🔑 Generate Hash (SHA256)
    │
    ▼
💾 Store in data_versions.json
    │
    ▼
🔄 Version Tracking
    │
    ├─── ✅ Integrity Verification
    ├─── 👥 Team Collaboration  
    └─── 🔬 Experiment Reproducibility
```

## 🔍 Integrity Verification Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              INTEGRITY VERIFICATION                            │
└─────────────────────────────────────────────────────────────────┘

📁 Current Dataset File
    │
    ▼
🔍 Calculate Current Hash
    │
    ▼
🔑 Compare with Expected Hash
    │
    ▼
    ├─── ✅ MATCH → Proceed with Experiment
    └─── ❌ MISMATCH → Stop Process
```

## 👥 Team Collaboration

```
┌─────────────────────────────────────────────────────────────────┐
│                TEAM COLLABORATION                              │
└─────────────────────────────────────────────────────────────────┘

👤 Team Member A
    │
    ▼
📤 Share: Dataset File + Expected Hash
    │
    ▼
👤 Team Member B
    │
    ▼
🔍 Verify Integrity
    │
    ▼
✅ Use Verified Dataset
    │
    ▼
🤖 Train Model
    │
    ▼
📊 Reproduce Results
```

## 🔬 Experiment Reproducibility

```
┌─────────────────────────────────────────────────────────────────┐
│            EXPERIMENT REPRODUCIBILITY                          │
└─────────────────────────────────────────────────────────────────┘

📋 Experiment Metadata
    │
    ▼
🔑 Load Expected Hash
    │
    ▼
📁 Locate Dataset File
    │
    ▼
✅ Verify Integrity
    │
    ▼
🤖 Run Experiment
    │
    ▼
📊 Reproduce Results
```

## 📊 Data Version Registry Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    data_versions.json                          │
└─────────────────────────────────────────────────────────────────┘

{
  "9fd78ddf1b085c9d...": {
    "dataset_name": "emotion_classification_data",
    "file_path": "data/raw/emotion_dataset.csv",
    "content_hash": "9fd78ddf1b085c9dc1caa5451570e12e497ad9360fa6bd00f6e25d300d212203",
    "description": "Initial version of emotion classification dataset",
    "tags": ["emotion", "audio", "classification"],
    "created_at": "2025-08-01T15:23:54.265372",
    "file_size": 4153
  }
}
```

## 🎯 Key Benefits

```
┌─────────────────────────────────────────────────────────────────┐
│                        BENEFITS                                │
└─────────────────────────────────────────────────────────────────┘

✅ Data Integrity
   ├── Prevents silent corruption
   ├── Ensures reproducibility
   └── Provides audit trail

✅ Team Collaboration
   ├── Centralized versioning
   ├── Easy sharing
   └── Consistent data

✅ MLOps Best Practices
   ├── Model versioning
   ├── Experiment tracking
   └── Deployment safety

✅ Compliance & Audit
   ├── Data lineage
   ├── Reproducibility
   └── Audit trail
```

## 🚀 Real-World Usage

```
┌─────────────────────────────────────────────────────────────────┐
│                    USAGE SCENARIOS                             │
└─────────────────────────────────────────────────────────────────┘

1. 📁 New Dataset Creation
   ├── Create/obtain dataset file
   ├── Call dvc.version_dataset()
   ├── Store returned hash as 'expected_hash'
   └── Use hash in experiments

2. 🔬 Experiment Reproducibility
   ├── Load expected hash from experiment metadata
   ├── Call dvc.verify_dataset_integrity()
   ├── Only proceed if verification passes
   └── Run experiment with verified dataset

3. 👥 Team Collaboration
   ├── Share dataset file with team
   ├── Share expected hash with team
   ├── Each team member verifies integrity
   └── All use identical dataset version

4. 🚀 Model Deployment
   ├── Store dataset hash with model metadata
   ├── Verify dataset integrity before deployment
   ├── Deploy model with verified data
   └── Track data lineage for audit
```

## 🔑 Core DVC Functions

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORE FUNCTIONS                              │
└─────────────────────────────────────────────────────────────────┘

📊 version_dataset(file_path, name, description)
   ├── Reads dataset file
   ├── Calculates SHA256 hash
   ├── Stores metadata in data_versions.json
   └── Returns hash (expected_hash)

🔍 verify_dataset_integrity(file_path, expected_hash)
   ├── Reads current file
   ├── Calculates current hash
   ├── Compares with expected_hash
   └── Returns True/False

📋 list_datasets()
   ├── Reads data_versions.json
   ├── Returns all versioned datasets
   └── Includes metadata for each

🔍 get_dataset_info(hash)
   ├── Looks up hash in data_versions.json
   ├── Returns complete metadata
   └── Includes file path, description, tags
```

## 📈 Integration with MLOps Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                MLOPS INTEGRATION                               │
└─────────────────────────────────────────────────────────────────┘

📊 Data Pipeline
    │
    ▼
🔍 DVC Versioning
    │
    ▼
🤖 Model Training
    │
    ▼
📋 Model Registry
    │
    ▼
🚀 Model Deployment
    │
    ▼
📊 Monitoring
    │
    ▼
🔄 Data Drift Detection
    │
    ▼
📈 Retraining Trigger
    │
    ▼
📊 Data Pipeline (loop)
```

---

*This simple diagram shows the core DVC workflow and how it integrates with the Speaking Feedback Tool for robust data versioning and integrity verification.* 