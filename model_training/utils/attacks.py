import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.detector import gram_margin_loss, ScoreDetector


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
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

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
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx


#class PGDTarget(nn.Module):
#    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
#        super().__init__()
#        self.epsilon = epsilon
#        self.num_steps = num_steps
#        self.step_size = step_size
#        self.grad_sign = grad_sign
#
#    def forward(self, model, bx, by):
#        """
#        :param model: the classifier's forward method
#        :param bx: batch of images
#        :param by: true labels
#        :return: perturbed batch of images
#        """
#        adv_bx = bx.detach()
#        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
#
#        random_by = np.random.randint(0, 1000, by.size(0))
#        for i in range(len(random_by)):
#            while random_by[i] == by[i]:
#                random_by[i] = np.random.randint(1000)
#        random_by = torch.LongTensor(random_by).cuda()
#
#        for i in range(self.num_steps):
#            adv_bx.requires_grad_()
#            with torch.enable_grad():
#                logits = model(adv_bx * 2 - 1)
#                loss = -F.cross_entropy(logits, random_by, reduction='sum')
#            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
#
#            if self.grad_sign:
#                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
#            else:
#                grad = normalize_l2(grad.detach())
#                adv_bx = adv_bx.detach() + self.step_size * grad
#
#            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)
#
#        return adv_bx

# class PGD(nn.Module):
#     def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
#         super().__init__()
#         self.epsilon = epsilon
#         self.num_steps = num_steps
#         self.step_size = step_size
#         self.grad_sign = grad_sign
#
#     def forward(self, model, bx, by):
#         """
#         :param model: the classifier's forward method
#         :param bx: batch of images
#         :param by: true labels
#         :return: perturbed batch of images
#         """
#         adv_bx = (bx + (torch.rand(bx.size()).cuda() - 0.5) * 2 * self.epsilon).clamp(0, 1).requires_grad_()
#
#         for i in range(self.num_steps):
#             logits = model(adv_bx*2 - 1)
#
#             loss = F.cross_entropy(logits, by, reduction='sum')
#
#             grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
#             if self.grad_sign:
#                 adv_bx = adv_bx + self.step_size * torch.sign(grad)
#             else:
#                 grad = normalize_l2(grad)  # normalize search speed
#                 adv_bx = adv_bx + self.step_size * grad
#             adv_bx = tensor_clamp(adv_bx, bx - self.epsilon, bx + self.epsilon).clamp(0, 1)
#             adv_bx = adv_bx.data.requires_grad_()
#
#         return adv_bx.clamp(0, 1)

class PGD_score(nn.Module):
    def __init__(self, epsilon=8./255, num_steps=10, step_size=2./255, score_scale=100, max_score=0.12, verbose=False, detector=ScoreDetector()):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.verbose = verbose
        self.detector = detector
        self.score_scale = score_scale
        self.max_score = max_score

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
                logits, feats_adv = model.gram_forward(adv_bx * 2 - 1)
                gram_score = F.softplus(self.detector.score(feats_adv) - self.max_score, beta=1000)
                cent_loss = F.cross_entropy(logits, by, reduction='mean').cuda()
                
                loss = cent_loss - self.score_scale * gram_score

                if self.verbose:
                    print("Step: {}, Cent: {}, Gram: {}, Total Loss: {}".format(i, cent_loss.data, gram_score.data, loss.data))
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)
            
        return adv_bx

class PGD_Gram(nn.Module):
    def __init__(self, detector, epsilon=8/255, num_steps=10, step_size=2/255, grad_sign=True, verbose=True):
        super().__init__()
        self.detector = detector
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.verbose = verbose
        
        
    
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
                logits, feats = model.gram_forward(adv_bx * 2 - 1)
                
                cent_loss = 0.0 * F.cross_entropy(logits, by, reduction='mean')
                gram_loss = self.detector.gram_loss(logits, feats)
                
                loss = cent_loss - gram_loss
                                
            if self.verbose:
                print("Step: {}, Cent: {}, Gram: {}, Total Loss: {}".format(i, cent_loss, gram_loss, loss))
            
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx

class PGD_margin(nn.Module):
    def __init__(self, epsilon=8./255, num_steps=10, step_size=2./255, margin = 20, margin_scale=1.0, verbose=False):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.verbose = verbose
        self.margin = margin
        self.margin_scale = margin_scale

    def forward(self, model, bx, by, feats_reg):
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
                logits, feats_adv = model.gram_forward(adv_bx * 2 - 1)
                gram_margin = self.margin_scale * F.softplus(self.detector.score(feats_adv) - .125, beta=100).cuda()
                cent_loss = F.cross_entropy(logits, by, reduction='mean').cuda()
                
                loss = cent_loss + gram_margin
                
                if self.verbose:
                    print("Step: {}, Cent: {}, Margin Loss: {}, Total Loss: {}".format(i, cent_loss, gram_margin, loss))
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)
            
        return adv_bx

class PGD_l2(nn.Module):
    def __init__(self, epsilon, num_steps, step_size):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

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
            logits = model(adv_bx * 2 - 1)

            loss = F.cross_entropy(logits, by, reduction='sum')

            grad = normalize_l2(torch.autograd.grad(loss, adv_bx, only_inputs=True)[0])
            adv_bx = adv_bx + self.step_size * grad
            adv_bx = tensor_clamp_l2(adv_bx, bx, self.epsilon).clamp(0, 1)
            adv_bx = adv_bx.data.requires_grad_()

        return adv_bx
