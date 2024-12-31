# Advanced-DMT
Code release for the paper "Advanced Discriminative Co-Saliency and Background Mining Transformer for Co-Salient Object Detection" by Long Li, Huichao Xie, Nian Liu, Dingwen Zhang, Rao Muhammad Anwer, Hisham Cholakkal, and Junwei Han.

![avatar](framework.jpg)

## Abstract
Most existing CoSOD models focus solely on extracting co-saliency cues while neglecting explicit exploration of background regions, potentially leading to difficulties in handling interference from complex background areas. To address this, this
paper proposes a Discriminative co-saliency and background Mining Transformer framework (DMT) to explicitly mine both co-saliency and background information and effectively model their discriminability. DMT first learns two types of tokens by disjointly extracting co-saliency and background information from segmentation features, then performs discriminability within the segmentation features guided by these well-learned tokens. In the first phase, we propose economic multi-grained correlation modules for efficient detection information extraction, including Region-to-Region (R2R), Contrast-induced Pixel-to-Token (CtP2T), and Co-saliency Token-to-Token (CoT2T) correlation modules. In the subsequent phase, we introduce Token-Guided Feature Refinement (TGFR) modules to enhance discriminability within the segmentation features. To further enhance the discriminative modeling and practicality of DMT, we first upgrade the original TGFR’s intra-image modeling approach to an intra-group one, thus proposing Group TGFR (G-TGFR), which is more suitable for the co-saliency task. Subsequently, we designed a Noise Propagation Suppression (NPS) mechanism to apply our model to a more practical open-world scenario, ultimately presenting our extended version, i.e. DMT+O. Extensive experimental results on both conventional CoSOD and open-world CoSOD benchmark datasets demonstrate the effectiveness of our proposed model.

## Training model
1. Download the pretrained VGG model from [Baidu Driver](https://pan.baidu.com/s/173-1VToeumXZy90cRw-Yqw)(sqd5) and put it into `./checkpoints` folder.
2. Run `python train.py`. 
3. The trained models with satisfactory performance will be saved in `./checkpoints/CONDA/`.

## Testing model
1. Download our trained model from [DUTS+CoCo9k](https://pan.baidu.com/s/1udfmF2xZHKO8qmUEc2YIgQ?pwd=qmbu) (qmbu) or [DUTS+CoCoSeg]( https://pan.baidu.com/s/1wYlUAlkUa2eFRd7B9gjz5A?pwd=2r6c) (2r6c) or [DUTS](https://pan.baidu.com/s/1eif2ch31qXg-ysuFDKa-gw?pwd=eq5n) (eq5n) and put them into `./Models` folder.
3. Run `python test.py`.
4. The prediction images will be saved in `./Prediction`. 
5. Run `python ./evaluation/eval_from_imgs.py` to evaluate the predicted results on three datasets and the evaluation scores will be written in `./evaluation/result`.
