import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import transforms

def pipeline_batch(bxs):
    pil = transforms.ToPILImage()
    return torch.squeeze(torch.stack([transform_test(pil(bx)) for bx in bxs]), dim=1)

def tensor_clamp(x, a_min, a_max):
    """
    like torch.clamp, except with bounds defined as tensors
    """
    out = torch.clamp(x - a_max, max=0) + a_max
    out = torch.clamp(out - a_min, min=0) + a_min
    return out


def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm


def tensor_clamp_l2(x, center, radius):
    """batched clamp of x into l2 ball around center of given radius"""
    x = x.data
    diff = x - center
    diff_norm = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)
    project_select = diff_norm > radius
    if project_select.any():
        diff_norm = diff_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        new_x = x
        new_x[project_select] = (center + (diff / diff_norm) * radius)[project_select]
        return new_x
    else:
        return x

class PGD(nn.Module):
    def __init__(self, epsilon=8/255, num_steps=10, step_size=2/255, grad_sign=True, mean = None, std = None):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        
        if mean is None:
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).cuda()
        else:
            self.mean = torch.FloatTensor(mean).view(1,3,1,1).cuda()
        if std is None:
            self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).cuda()
        else:
            self.std = torch.FloatTensor(std).view(1,3,1,1).cuda()

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model((adv_bx - self.mean)/self.std)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx

class PGD_l2(nn.Module):
    def __init__(self, epsilon=1.0, num_steps=10, step_size=0.2, mean = None, std=None):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        if mean is None:
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).cuda()
        else:
            self.mean = torch.FloatTensor(mean).view(1,3,1,1).cuda()
        if std is None:
            self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).cuda()
        else:
            self.std = torch.FloatTensor(std).view(1,3,1,1).cuda()
    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        init_noise = normalize_l2(torch.randn(bx.size()).cuda()) * np.random.rand() * self.epsilon
        adv_bx = (bx + init_noise).clamp(0, 1).requires_grad_()

        for i in range(self.num_steps):
            logits = model((adv_bx - self.mean)/self.std)

            loss = F.cross_entropy(logits, by, reduction='sum')

            grad = normalize_l2(torch.autograd.grad(loss, adv_bx, only_inputs=True)[0])
            adv_bx = adv_bx + self.step_size * grad
            adv_bx = tensor_clamp_l2(adv_bx, bx, self.epsilon).clamp(0, 1)
            adv_bx = adv_bx.data.requires_grad_()

        return adv_bx
