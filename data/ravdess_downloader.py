"""
RAVDESS Dataset Downloader and Preprocessor
Downloads and prepares RAVDESS dataset for emotion recognition training
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAVDESSDownloader:
    """Download and preprocess RAVDESS dataset"""
    
    def __init__(self, data_dir: str = "data/ravdess"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # RAVDESS dataset information
        self.dataset_info = {
            "name": "RAVDESS",
            "description": "Ryerson Audio-Visual Database of Emotional Speech and Song",
            "url": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
            "filename": "Audio_Speech_Actors_01-24.zip",
            "emotions": {
                "01": "neutral",
                "02": "calm", 
                "03": "happy",
                "04": "sad",
                "05": "angry",
                "06": "fearful",
                "07": "disgust",
                "08": "surprised"
            },
            "intensities": {
                "01": "normal",
                "02": "strong"
            },
            "statements": {
                "01": "Kids are talking by the door",
                "02": "Dogs are sitting by the door"
            },
            "repetitions": {
                "01": "1st repetition",
                "02": "2nd repetition"
            }
        }
    
    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download RAVDESS dataset
        
        Args:
            force_download (bool): Force re-download even if file exists
            
        Returns:
            bool: True if successful
        """
        zip_path = self.data_dir / self.dataset_info["filename"]
        
        if zip_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {zip_path}")
            return True
        
        logger.info(f"Downloading RAVDESS dataset from {self.dataset_info['url']}")
        
        try:
            response = requests.get(self.dataset_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Download completed: {zip_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def extract_dataset(self) -> bool:
        """
        Extract downloaded dataset
        
        Returns:
            bool: True if successful
        """
        zip_path = self.data_dir / self.dataset_info["filename"]
        extract_dir = self.data_dir / "extracted"
        
        if not zip_path.exists():
            logger.error(f"Dataset file not found: {zip_path}")
            return False
        
        logger.info(f"Extracting dataset to {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info("Extraction completed")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse RAVDESS filename to extract metadata
        
        Args:
            filename (str): RAVDESS filename
            
        Returns:
            Dict: Parsed metadata
        """
        # RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        # Example: 03-01-01-01-01-01-01.wav
        
        parts = filename.replace('.wav', '').split('-')
        
        if len(parts) != 7:
            return {}
        
        return {
            "modality": parts[0],
            "vocal_channel": parts[1], 
            "emotion": self.dataset_info["emotions"].get(parts[2], "unknown"),
            "intensity": self.dataset_info["intensities"].get(parts[3], "unknown"),
            "statement": self.dataset_info["statements"].get(parts[4], "unknown"),
            "repetition": self.dataset_info["repetitions"].get(parts[5], "unknown"),
            "actor": parts[6],
            "filename": filename
        }
    
    def create_dataset_manifest(self) -> pd.DataFrame:
        """
        Create dataset manifest with metadata
        
        Returns:
            pd.DataFrame: Dataset manifest
        """
        extract_dir = self.data_dir / "extracted"
        
        if not extract_dir.exists():
            logger.error(f"Extracted directory not found: {extract_dir}")
            return pd.DataFrame()
        
        manifest_data = []
        
        # Find all wav files
        for wav_file in extract_dir.rglob("*.wav"):
            metadata = self.parse_filename(wav_file.name)
            if metadata:
                metadata["filepath"] = str(wav_file)
                metadata["duration"] = None  # Will be filled later
                manifest_data.append(metadata)
        
        manifest_df = pd.DataFrame(manifest_data)
        
        if not manifest_df.empty:
            logger.info(f"Created manifest with {len(manifest_df)} files")
            manifest_df.to_csv(self.data_dir / "manifest.csv", index=False)
        
        return manifest_df
    
    def get_audio_features(self, filepath: str) -> Dict[str, float]:
        """
        Extract basic audio features
        
        Args:
            filepath (str): Path to audio file
            
        Returns:
            Dict: Audio features
        """
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(filepath, sr=None)
            
            # Extract features
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "rms_energy": np.sqrt(np.mean(y**2)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                "mfcc_1": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[0]),
                "mfcc_2": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1]),
                "mfcc_3": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[2])
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Could not extract features from {filepath}: {e}")
            return {"duration": 0.0}
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training
        
        Returns:
            Tuple: (features_df, labels_df)
        """
        manifest_df = self.create_dataset_manifest()
        
        if manifest_df.empty:
            logger.error("No data found in manifest")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract features
        features_list = []
        for _, row in manifest_df.iterrows():
            features = self.get_audio_features(row["filepath"])
            features.update({
                "emotion": row["emotion"],
                "intensity": row["intensity"],
                "actor": row["actor"],
                "filepath": row["filepath"]
            })
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Create labels
        labels_df = features_df[["emotion", "intensity", "actor", "filepath"]].copy()
        
        # Save processed data
        features_df.to_csv(self.data_dir / "features.csv", index=False)
        labels_df.to_csv(self.data_dir / "labels.csv", index=False)
        
        logger.info(f"Prepared training data: {len(features_df)} samples")
        logger.info(f"Emotion distribution: {features_df['emotion'].value_counts().to_dict()}")
        
        return features_df, labels_df
    
    def get_dataset_info(self) -> Dict:
        """
        Get dataset information
        
        Returns:
            Dict: Dataset information
        """
        manifest_path = self.data_dir / "manifest.csv"
        features_path = self.data_dir / "features.csv"
        
        info = {
            "data_dir": str(self.data_dir),
            "manifest_exists": manifest_path.exists(),
            "features_exists": features_path.exists(),
            "dataset_info": self.dataset_info
        }
        
        if manifest_path.exists():
            manifest_df = pd.read_csv(manifest_path)
            info["total_files"] = len(manifest_df)
            info["emotion_distribution"] = manifest_df["emotion"].value_counts().to_dict()
        
        if features_path.exists():
            features_df = pd.read_csv(features_path)
            info["total_samples"] = len(features_df)
            info["feature_columns"] = list(features_df.columns)
        
        return info


def main():
    """Main function to download and prepare RAVDESS dataset"""
    downloader = RAVDESSDownloader()
    
    # Download dataset
    if downloader.download_dataset():
        # Extract dataset
        if downloader.extract_dataset():
            # Prepare training data
            features_df, labels_df = downloader.prepare_training_data()
            
            if not features_df.empty:
                print("✅ RAVDESS dataset prepared successfully!")
                print(f"📊 Total samples: {len(features_df)}")
                print(f"🎭 Emotions: {features_df['emotion'].value_counts().to_dict()}")
            else:
                print("❌ Failed to prepare training data")
        else:
            print("❌ Failed to extract dataset")
    else:
        print("❌ Failed to download dataset")


if __name__ == "__main__":
    main() 