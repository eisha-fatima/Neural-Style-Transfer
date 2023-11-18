import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_loader(img_path: str, transforms: transforms, device: str | torch.device):
    """
    Reads image from path applies transforms and adds a batch dimension.
    """

    img = Image.open(img_path)
    img = transforms(img).unsqueeze(0)
    return img.to(device, torch.float)


def imshow(tensor, transform, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transform(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


def train(
    model: nn.Module,
    style_losses: list,
    content_losses: list,
    input_img: torch.Tensor,
    optimizer: optim,
    num_steps=300,
    style_weight=100000,
    content_weight=1,
):
    """
    Fits the input image to incorporate both style and content of the provided images for a given number of steps.
    Computes loss and backpropogates to modify pixel values of input image till it converges close to 0 loss.

    Returns:
        result (PIL.Image) : Style transferred image.
    """

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    runs = [0]
    while runs[0] < num_steps:

        def closure():

            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0
            for s1 in style_losses:
                style_score += s1.loss
            for c1 in content_losses:
                content_score += c1.loss

            style_score = style_score * style_weight
            content_score = content_score * content_weight

            loss = style_score + content_score
            loss.backward()

            if runs[0] % 10 == 0:
                print(
                    f'STEP {runs[0]}: Content Loss: {content_score} , Style Loss: {style_score}')
            runs[0] += 1

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def create_colored_mask(content_img, mask_img) -> np.ndarray:
    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(content_img, content_img, mask=mask_inv)

    # Create a black image with the same size as the original image
    black_img = np.zeros_like(content_img)

    # Fill the black image with black color in the areas corresponding to the mask
    black_img[mask != 0] = (0, 0, 0)

    # Combine the masked original image with the black image
    result = cv2.add(masked_img, black_img)

    return result


def postprocess(generated_img: np.ndarray, masked_img: np.ndarray):
    """
    Paste backs masked image back onto the generated image.
    Both generated image and masked image should be of the same size.

    Args:
        generated_img (np.ndarray): Numpy array of style transferred images.
        masked_img (np.ndarray): Numpy array of masked image. All black except portion that is to be pasted back.

    Returns:
        result (np.ndarray) : Numpy array of same shape as input images.  
    """

    # Create a mask from the masked image by thresholding it
    gray_mask = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Extract the region of interest (ROI) from the original image
    roi = cv2.bitwise_and(generated_img, generated_img, mask=mask_inv)

    # Extract the region of interest from the masked image
    masked_roi = cv2.bitwise_and(masked_img, masked_img, mask=mask)

    # Combine the ROI from the original image and the masked ROI
    result = cv2.add(roi, masked_roi)

    return result
