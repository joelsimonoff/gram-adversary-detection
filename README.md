# Gram Matrix Adversarial Example Detector

This repository is dedicated to exploring the use of gram matrices for adversarial detection.

Main Attributions:
- Gram Matrix Out Of Distribution Detector (good chunk of source code is from this repo): https://github.com/VectorInstitute/gram-ood-detection
- Gram matrices were made popular via the Neural Style Transfer paper: https://arxiv.org/pdf/1508.06576.pdf

## Summary Of New Work

We used the Vector Institute's paper *(VI detector)* on utilizing gram matrices to detect out of distribution examples, https://arxiv.org/abs/1912.12510, as a jumping off point for an adversarial example detector.

### Out of Distribution (OOD) vs Adversarial

OOD examples are images that the model is not trained to predict on. For example, ImageNet images are out of distribution for a CIFAR10 model. Adversarial examples are images with small perturbations designed to fool models. We are using $L_\infty$ adversarial examples which means that $\lVert \text{Img} - \text{(Img + Pertubation)} \rVert_\infty < \epsilon$.

### Minimum Viable Detector

In order to repurpose the detector in the most effective way possible, we need to see what to remove from the VI detector and what we may need to add. We found that we did not need to take the gram matrices to a specific power and we do not need to look at the gram matrices from every layer. Looking at the matrices from shallower layers (closer to input) seems to yield better adversarial detection accuracy. We found that in some circumstances we gained accuracy by setting `powers` from the VI detector to `powers=[1/2, 1]`; however, the benefit is minimal at best thus we recommend removing the powers variable.

Gram Matrix Calculator VI Detector:

```python
def G_p(ob, p):
    temp = ob.detach()

    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2)
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)

    return temp
```

Our Gram Calculator (no powers):

```python
def G_p(temp):
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2)
    return temp.reshape(temp.shape[0],-1)
```

_Note: we found that summing across dim2 seems to accurately represent the total correlation of a feature with the other features._

For this method, we leave in calibration code that calculates the max/min values of the training set for each layer in the model. This calibration step is slow and bulky. In order to remove the calibration step we used 2 techniques (will still need to tune a threshold, but less intensive than the prior detector):

1. **Margin Detector:** Calculating a margin between adversarial perturbed examples and their corresponding test set examples. This works well for training loops because we'll often have both an adversarial example and its corresponding original image. However, this cannot work as a practical detector because at test time we don't have access to non-adversarial perturbed images in addition to the corresponding adversarial examples. The benefit of training in the loop with this detector is that it will force adversarial perturbations to increase the magnitude of the gram matrices (increases detectability). Unfortunately, it does this by obfuscating the gradient of the PGD margin loss with respect to the input (with each PGD step the attack gets worse which means the adversarial loss is not smooth).

2. **Score Detector:** This detector is based on the original style loss presented in the neural style transfer paper: https://arxiv.org/pdf/1508.06576.pdf:

```python
def G_p(self, temp):
    normalizer = torch.prod(torch.tensor(temp.shape[1:]))
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = (torch.matmul(temp,temp.transpose(dim0=2,dim1=1))).reshape(temp.shape[0],-1).sum(dim=1)

    return temp/normalizer
```

We then take the average value of the gram vector for each layer. At the end we average the value for each layer to assign a score. The score tends to be higher for adversarial examples than for regular examples. **With this method we see an AUROC in the range of 0.94.** The code for this detector can be found in the `ScoreDetector` class in `new-work/model-training/utils/detector.py`.


### Adversaries

For each detector (including the original detector that requires calibration), we need an adversary that can create adversarial examples that fool the detector and the model (adversaries that render the detector ineffective). We created these adversaries in `new-work/model-training/utils/attacks.py`. Most take the form:

$\text{min}_{\text{detectability}} \text{max}_{\text{cross_entropy}} \text{cross_entropy} + \text{detectability}$

Essentially: we want to maximize the cross_entropy (encourages the adversarial example to fool the model), and we want to minimize the detectability (making it hard for the detector to spot the example).

### Training

In order for a model to be consider robust it must meet one of the following two scenarios:

1. It is attacked, but correctly classifies adversarial examples. This is the motivation behind adversarial training.
2. The attack is detected. If the attack is detected, then the system knows not to trust the output of the model.

We attempt to train the model so it improves the detectability of adversarial examples. The goal is that if we make it near impossible to trick the model and the detector at the same time then the attack is effectively useless. Thus our training loss is of the form:

$\text{min}_{\text{cross_entropy}} \text{max}_{\text{detectability}} \text{cross_entropy} + \text{detectability}$

This is essentially the inverse of the loss used by the attacker.

_Note: this is not a GAN because the attacker is a fixed algorithm and not trainable, thus we are only training the model._

## Project Structure

### original-gram-ood-detection

This directory contains code from https://github.com/VectorInstitute/gram-ood-detection that is unmodified. This code is included because it can be helpful to look at for reference.

### new-work

This directory contains our work.

**Top Level**

Jupyter notebooks with experimental code. This code was mainly used to conduct ablation tests and general experimentation.

**model-training:**

- `gram_pgd_scratch.ipynb`: used to prototype new adversarial attacks designed to beat the detector.
- `WRNTraining_Master.ipynb`: training a model that is resistant to attacks that fool the score detector.
- `WRNTraining_Margin.ipynb`: training a model that is resistant to attack that fools the margin detector.
- `utils/attacks.py`: contains PGD based attacks with custom loss functions to fool the detector.
- `utils/detector.py`: contains detector code.
- `utils/wrn.py`: PyTorch WRN model.
- `utils/calculate_log.py`: detector metrics (mostly from VectorInstitute code).
- `benchmark_ckpts`: normally trained wrn CIFAR10 models to use as a benchmark.
- `checkpoints_margin_01`: checkpoints for wrn model designed to disable attackers from tricking the margin based detector.
- `checkpoints_margin_01_v2`: similar to `checkpoints_margin_01` except the attacker for this model will only minimize the margin loss to a point and then focus on maximizing cross_entropy.
- `checkpoints_score_v1`: designed to disable attacker from tricking the score detector.
