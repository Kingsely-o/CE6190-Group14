"""
Evaluation metrics for image generation quality assessment
Implements: FID, CLIP Score, Inception Score, and LPIPS
"""

import os
from pathlib import Path
from typing import List, Union, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm
import open_clip

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import FID_BATCH_SIZE, FID_DIMS, IS_SPLITS, DEVICE
from utils.helpers import Timer, ProgressTracker

warnings.filterwarnings('ignore')


# ============================================================================
# FID (Frechet Inception Distance)
# ============================================================================

class InceptionFeatureExtractor:
    """Extract features from Inception V3 for FID calculation."""

    def __init__(self, device: str = DEVICE):
        """Initialize Inception V3 model."""
        self.device = device
        print("[METRIC] Loading Inception V3 for FID...")

        # Load pretrained Inception V3
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.model.fc = torch.nn.Identity()  # Remove final classification layer
        self.model = self.model.to(device)
        self.model.eval()

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, images: List[Image.Image], batch_size: int = FID_BATCH_SIZE) -> np.ndarray:
        """
        Extract features from a list of images.

        Args:
            images: List of PIL Images
            batch_size: Batch size for processing

        Returns:
            Feature array of shape (num_images, 2048)
        """
        features = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]

                # Preprocess images
                batch_tensor = torch.stack([self.preprocess(img) for img in batch])
                batch_tensor = batch_tensor.to(self.device)

                # Extract features
                batch_features = self.model(batch_tensor)
                features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)

    def extract_features_from_dir(self, image_dir: Union[str, Path], batch_size: int = FID_BATCH_SIZE) -> np.ndarray:
        """
        Extract features from all images in a directory.

        Args:
            image_dir: Directory containing images
            batch_size: Batch size for processing

        Returns:
            Feature array
        """
        image_dir = Path(image_dir)
        image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        print(f"[FID] Extracting features from {len(image_paths)} images...")
        images = [Image.open(p).convert("RGB") for p in tqdm(image_paths, desc="Loading images")]

        return self.extract_features(images, batch_size)


def calculate_fid_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and covariance of features.

    Args:
        features: Feature array

    Returns:
        Tuple of (mean, covariance)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                               mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Calculate Frechet distance between two multivariate Gaussians.

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution

    Returns:
        Frechet distance
    """
    # Calculate squared difference of means
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical errors
    if not np.isfinite(covmean).all():
        print("[WARNING] FID calculation produced non-finite values")
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Handle complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_fid(generated_images_dir: Union[str, Path],
                 real_images_dir: Union[str, Path],
                 batch_size: int = FID_BATCH_SIZE,
                 device: str = DEVICE) -> float:
    """
    Calculate FID score between generated and real images.

    Args:
        generated_images_dir: Directory with generated images
        real_images_dir: Directory with real images
        batch_size: Batch size for feature extraction
        device: Device to use for computation

    Returns:
        FID score (lower is better)
    """
    print("\n[FID] Calculating Frechet Inception Distance...")

    with Timer("FID calculation"):
        # Initialize feature extractor
        extractor = InceptionFeatureExtractor(device)

        # Extract features from both sets
        print("[FID] Extracting features from generated images...")
        generated_features = extractor.extract_features_from_dir(generated_images_dir, batch_size)

        print("[FID] Extracting features from real images...")
        real_features = extractor.extract_features_from_dir(real_images_dir, batch_size)

        # Calculate statistics
        print("[FID] Calculating statistics...")
        mu_gen, sigma_gen = calculate_fid_statistics(generated_features)
        mu_real, sigma_real = calculate_fid_statistics(real_features)

        # Calculate FID
        fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

    print(f"[FID] Score: {fid_score:.2f}")
    return fid_score


# ============================================================================
# CLIP Score
# ============================================================================

class CLIPScorer:
    """Calculate CLIP score for text-image alignment."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = DEVICE):
        """Initialize CLIP model."""
        self.device = device
        print(f"[METRIC] Loading CLIP model: {model_name}")

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name.replace('/', '-'),
            pretrained='openai'
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_name.replace('/', '-'))

    def calculate_clip_score(self, images: List[Image.Image], prompts: List[str]) -> float:
        """
        Calculate CLIP score between images and prompts.

        Args:
            images: List of PIL Images
            prompts: List of text prompts

        Returns:
            Average CLIP score
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")

        scores = []

        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                # Preprocess image
                image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                # Tokenize text
                text_tokens = self.tokenizer([prompt]).to(self.device)

                # Get embeddings
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Calculate similarity
                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)

        return float(np.mean(scores))

    def calculate_clip_scores_list(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        """
        Calculate individual CLIP scores for each image-prompt pair.

        Args:
            images: List of PIL Images
            prompts: List of text prompts

        Returns:
            List of CLIP scores
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")

        scores = []

        with torch.no_grad():
            for img, prompt in tqdm(zip(images, prompts), total=len(images), desc="Computing CLIP scores"):
                # Preprocess image
                image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                # Tokenize text
                text_tokens = self.tokenizer([prompt]).to(self.device)

                # Get embeddings
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Calculate similarity
                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)

        return scores


def calculate_clip_score(images: List[Image.Image], prompts: List[str],
                        model_name: str = "ViT-B/32", device: str = DEVICE) -> float:
    """
    Calculate CLIP score for text-image alignment.

    Args:
        images: List of PIL Images
        prompts: List of text prompts corresponding to images
        model_name: CLIP model to use
        device: Device to use for computation

    Returns:
        Average CLIP score (higher is better)
    """
    print("\n[CLIP] Calculating CLIP Score...")

    with Timer("CLIP score calculation"):
        scorer = CLIPScorer(model_name, device)
        clip_score = scorer.calculate_clip_score(images, prompts)

    print(f"[CLIP] Score: {clip_score:.4f}")
    return clip_score


# ============================================================================
# Inception Score
# ============================================================================

def calculate_inception_score(images: List[Image.Image],
                              batch_size: int = 32,
                              splits: int = IS_SPLITS,
                              device: str = DEVICE) -> Tuple[float, float]:
    """
    Calculate Inception Score.

    Args:
        images: List of PIL Images
        batch_size: Batch size for processing
        splits: Number of splits for IS calculation
        device: Device to use for computation

    Returns:
        Tuple of (mean IS, std IS)
    """
    print("\n[IS] Calculating Inception Score...")

    with Timer("Inception Score calculation"):
        # Load Inception V3
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model = model.to(device)
        model.eval()

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Get predictions
        preds = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_tensor = torch.stack([preprocess(img) for img in batch])
                batch_tensor = batch_tensor.to(device)

                pred = model(batch_tensor)
                pred = F.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Calculate IS
        split_scores = []
        split_size = len(preds) // splits

        for k in range(splits):
            part = preds[k * split_size:(k + 1) * split_size]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py)))
            split_scores.append(np.exp(np.mean(scores)))

        is_mean = float(np.mean(split_scores))
        is_std = float(np.std(split_scores))

    print(f"[IS] Score: {is_mean:.2f} ± {is_std:.2f}")
    return is_mean, is_std


# ============================================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================================

def calculate_lpips(images: List[Image.Image], device: str = DEVICE) -> float:
    """
    Calculate LPIPS diversity score.

    Args:
        images: List of PIL Images
        device: Device to use for computation

    Returns:
        Average LPIPS distance (higher = more diverse)
    """
    print("\n[LPIPS] Calculating LPIPS diversity...")

    try:
        import lpips
    except ImportError:
        print("[ERROR] lpips package not installed. Install with: pip install lpips")
        return 0.0

    with Timer("LPIPS calculation"):
        # Load LPIPS model
        loss_fn = lpips.LPIPS(net='alex').to(device)

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Calculate pairwise distances
        distances = []
        num_pairs = min(100, len(images) * (len(images) - 1) // 2)  # Limit pairs for efficiency

        with torch.no_grad():
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if len(distances) >= num_pairs:
                        break

                    img1 = preprocess(images[i]).unsqueeze(0).to(device)
                    img2 = preprocess(images[j]).unsqueeze(0).to(device)

                    dist = loss_fn(img1, img2).item()
                    distances.append(dist)

                if len(distances) >= num_pairs:
                    break

        lpips_score = float(np.mean(distances))

    print(f"[LPIPS] Diversity score: {lpips_score:.4f}")
    return lpips_score


# ============================================================================
# Utility Functions
# ============================================================================

def load_images_from_dir(image_dir: Union[str, Path]) -> List[Image.Image]:
    """
    Load all images from a directory.

    Args:
        image_dir: Directory containing images

    Returns:
        List of PIL Images
    """
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")

    return images


if __name__ == "__main__":
    print("Metrics module loaded successfully!")
    print("\nAvailable metrics:")
    print("- calculate_fid(generated_dir, real_dir)")
    print("- calculate_clip_score(images, prompts)")
    print("- calculate_inception_score(images)")
    print("- calculate_lpips(images)")
