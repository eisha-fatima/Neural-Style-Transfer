import torch
import torch.optim as optim

import torchvision.transforms as transforms

from model import VGG19
from utils import img_loader, train, imshow, postprocess, create_colored_mask

import os
import numpy as np
from PIL import Image
import cv2

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 299
    NUM_STEPS = 300

    unloader_transforms = transforms.ToPILImage()
    STYLE_IMG_PATH = 'images/styleImg1.jpg'
    CONTENT_IMG_PATH = 'images/contentImg2.JPG'
    MASK_IMG_PATH = 'images/mask2.png'
    SAVE_DIR = 'results/'
    os.makedirs(SAVE_DIR, exist_ok=True)

    img_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    style_img = img_loader(STYLE_IMG_PATH, img_transforms, DEVICE)
    content_img = img_loader(CONTENT_IMG_PATH, img_transforms, DEVICE)
    mask_img = cv2.resize(cv2.imread(MASK_IMG_PATH), (IMG_SIZE, IMG_SIZE))

    assert style_img.size() == content_img.size()

    # imshow(style_img, transform=unloader_transforms, title='Style Image')
    # imshow(content_img, transform=unloader_transforms, title='Content Image')

    # USING A COPY OF CONTENT IMAGE AS STATING INPUT IMAGE
    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img])

    vgg = VGG19(device=DEVICE)
    model, style_losses, content_losses = vgg.build_model(
        content_img=content_img, style_img=style_img)

    output = train(
        model,
        style_losses,
        content_losses,
        input_img,
        optimizer,
        num_steps=NUM_STEPS
    )

    # imshow(output, unloader_transforms, title='Output')
    result = unloader_transforms(output.cpu().squeeze(0))
    content_img = unloader_transforms(content_img.cpu().squeeze(0))

    result.save(os.path.join(SAVE_DIR, 'result_original.png'))
    result = Image.fromarray(postprocess(np.asarray(
        result), create_colored_mask(np.asarray(content_img), mask_img)))
    result.save(os.path.join(SAVE_DIR, 'result_postproccessed.png'))
