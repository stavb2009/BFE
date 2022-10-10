from pylab import *
import cv2
import torch
from albumentations.pytorch.transforms  import Compose, Normalize
from albumentations.pytorch.transforms import img_to_tensor
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    #return x.cuda(non_blocking=True) if torch.cuda.is_available() else x

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

def get_model(model_path, model_type='UNet11'):
    """
    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :return:
    """
    num_classes = 1
    problem_type = 'binary'

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img

def generate_mask(model_path,img_file_name, model_type,show=1):
    """
    :param model_path:
    :param img_file_name:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :return:
    """
    model = get_model(model_path, model_type=model_type)
    image = load_image(img_file_name)
    if show : imshow(image)
    with torch.no_grad():
        input_image = torch.unsqueeze(img_to_tensor(img_transform(p=1)(image=image)['image']).cuda(), dim=0)
    mask = model(input_image)
    mask_array = mask.data[0].cpu().numpy()[0]
    if show :
        imshow(mask_array > 0)
        imshow(mask_overlay(image, (mask_array > 0).astype(np.uint8)))
    return mask_array