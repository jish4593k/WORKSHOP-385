import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
import time

BATCH_SIZE = 64

def load_images_from_folder(folder):
    """
    Load images from a folder.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        np.ndarray: Array of images.
    """
    images = []
    for filename in tqdm.tqdm(os.listdir(folder), desc="Loading images", ncols=80):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = img.resize((299, 299))  # Resize images to fit Inception model input
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)

def inception_logits(images):
    """
    Get Inception model logits for images.

    Args:
        images (np.ndarray): Array of images.

    Returns:
        np.ndarray: Inception model logits.
    """
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    # Normalize images to match Inception model input
    images_tensor = F.interpolate(images_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    images_tensor /= 255.0

    with torch.no_grad():
        logits, _ = model(images_tensor)

    return logits.numpy()

def get_inception_probs(inps):
    """
    Get Inception probabilities for images.

    Args:
        inps (np.ndarray): Array of images.

    Returns:
        np.ndarray: Inception model probabilities.
    """
    preds = []
    n_batches = len(inps) // BATCH_SIZE
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        pred = inception_logits(inp)[:, :1000]
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits):
    """
    Calculate Inception Score from Inception probabilities.

    Args:
        preds (np.ndarray): Inception model probabilities.
        splits (int): Number of splits.

    Returns:
        Tuple[float, float]: Mean and standard deviation of Inception Score.
    """
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10):
    """
    Compute the Inception Score for a set of images.

    Args:
        images (np.ndarray): Array of images.
        splits (int): Number of splits.

    Returns:
        Tuple[float, float]: Mean and standard deviation of Inception Score.
    """
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(np.max(images[0]) <= 1)
    assert(np.min(images[0]) >= -1)

    start_time = time.time()
    preds = get_inception_probs(images)
    print(f'Inception Score for {preds.shape[0]} samples in {splits} splits')
    mean, std = preds2score(preds, splits)
    print(f'Inception Score calculation time: {time.time() - start_time} s')
    return mean, std

if __name__ == "__main__":
    # Example usage:
    folder_path = "path/to/images"
    images = load_images_from_folder(folder_path)
    mean_score, std_score = get_inception_score(images)
    print(f"Inception Score: Mean={mean_score}, Std={std_score}")
