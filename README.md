# Advanced-DMT
Code release for the paper "Advanced Discriminative Co-Saliency and Background Mining Transformer for Co-Salient Object Detection" by Long Li, Huichao Xie, Nian Liu, Dingwen Zhang, Rao Muhammad Anwer, Hisham Cholakkal, and Junwei Han.

![avatar](framework.jpg)

## Training model
1. Download the pretrained VGG model from [Baidu Driver](https://pan.baidu.com/s/173-1VToeumXZy90cRw-Yqw)(sqd5) and put it into `./checkpoints` folder.
2. Run `python train.py`. 
3. The trained models with satisfactory performance will be saved in `./checkpoints/CONDA/`.

## Testing model
1. Download our trained model from [DUTS+CoCo9k](https://pan.baidu.com/s/1udfmF2xZHKO8qmUEc2YIgQ?pwd=qmbu) (qmbu) or [DUTS+CoCoSeg]( https://pan.baidu.com/s/1wYlUAlkUa2eFRd7B9gjz5A?pwd=2r6c) (2r6c) or [DUTS](https://pan.baidu.com/s/1eif2ch31qXg-ysuFDKa-gw?pwd=eq5n) (eq5n) and put them into `./Models` folder.
3. Run `python test.py`.
4. The prediction images will be saved in `./Prediction`. 
5. Run `python ./evaluation/eval_from_imgs.py` to evaluate the predicted results on three datasets and the evaluation scores will be written in `./evaluation/result`.
