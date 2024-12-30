import os
import argparse
import traceback

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from config.config import get_cfg
from dataset import build_data_loader

from CoSODNet import CoSODNet
import transforms as trans
import cv2
import evaluation.metric as M

def get_metric_function():
    return {
        'FM': M.Fmeasure_and_FNR(),
        'WFM': M.WeightedFmeasure(),
        'SM': M.Smeasure(),
        'EM': M.Emeasure(),
        'MAE': M.MAE()
    }

def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser("CoSOD_Test", add_help=False)
    parser.add_argument("-config_file", default="./config/cosod.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("-num_works", default=1, type=int)
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-device_id", type=str, default="0")
    parser.add_argument("-img_size", type=int, default=256)
    parser.add_argument("-max_num", type=int, default=25)
    parser.add_argument("-model_root_dir", default="./Models")
    parser.add_argument("-test_data_root", type=str, default="./dataset/test_data")
    parser.add_argument("-test_datasets", nargs='+', default=["CoCA", "CoSal2015", "CoSOD3k"])
    parser.add_argument("-save_dir", type=str, default='./Predictions')
    parser.add_argument("-test_model_name", type=str, default="DC+C9.pth")
    return parser


def _get_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    return cfg


def test_group(model, group_data, save_root, max_num):
    img_num = group_data['imgs'].shape[1]
    groups = list(range(0, img_num + 1, max_num))
    if groups[-1] != img_num:
        groups.append(img_num)

    print(groups)

    for i in range(len(groups) - 1):
        if i == len(groups) - 2:
            end = groups[i + 1]
            start = max(0, end - max_num)
        else:
            start = groups[i]
            end = groups[i + 1]

        print(start, end)

        inputs = Variable(group_data['imgs'][:, start:end].squeeze(0).cuda())
        subpaths = group_data['subpaths'][start:end]
        ori_sizes = group_data['ori_sizes'][start:end]

        # img_name = '_'.join(subpaths[0][0][:-4].split('/')).replace(' ', '_')
        with torch.no_grad():

            result = model(inputs)

            co_preds = result.pop("co_pred")
            pred_prob = torch.sigmoid(co_preds)

            save_final_path = os.path.join(save_root, subpaths[0][0].split('/')[0])
            os.makedirs(save_final_path, exist_ok=True)

            for p_id in range(end - start):
                pre = pred_prob[p_id, :, :, :].data.cpu()

                subpath = subpaths[p_id][0]
                ori_size = (ori_sizes[p_id][1].item(),
                            ori_sizes[p_id][0].item())

                transform = trans.Compose([
                    trans.ToPILImage(),
                    trans.Scale(ori_size)
                ])
                outputImage = transform(pre)
                filename = subpath.split('/')[1]
                outputImage.save(os.path.join(save_final_path, filename))

def main(args):
    cfg = _get_cfg(args.config_file)
    model = CoSODNet(cfg)
    model.cuda()

    model_dir = os.path.join(args.model_root_dir, args.test_model_name)

    print(model_dir)
    model.load_state_dict(torch.load(model_dir))
    print("Model loaded from {}".format(model_dir))
    test_loaders = build_data_loader(args, mode='test')

    for dataset, data_loader in test_loaders.items():
        save_root = os.path.join(args.save_dir, dataset, '{}_iter{}.pth'.format(model_name, idx))
        print("testing on {}".format(dataset))

        for idx_, group_data in enumerate(data_loader):
            print('{}/{}'.format(idx_, len(data_loader)))

            max_num = args.max_num

            flag = True
            while flag:
                try:
                    test_group(model, group_data, save_root, max_num)
                    flag = False
                except Exception as e:
                    print("set max_num as {}".format(max_num - 2))
                    max_num = max_num - 1
                    print(e.args)
                    print(traceback.format_exc())
                    if max_num == 0:
                        break
                    continue

        dataset_list = args.test_datasets
        data_root = args.test_data_root
        pred_root = args.save_dir
        for i in range(len(dataset_list)):
            dataset = dataset_list[i]
            print('evaluating on {} dataset.'.format(dataset))

            pred_data_dir = os.path.join(pred_root, dataset)
            label_data_dir = os.path.join(data_root, dataset, 'GroundTruth')

            log_file = open('./evaluation/result/{}.txt'.format(dataset), 'a')

            mertic_fun = get_metric_function()

            classes = os.listdir(label_data_dir)
            for k in range(len(classes)):
                print('\r{}/{}'.format(k, len(classes)), end="", flush=True)
                class_name = classes[k]
                img_list = os.listdir(os.path.join(label_data_dir, class_name))
                for l in range(len(img_list)):
                    img_name = img_list[l]
                    # print("{}/{}".format(class_name, img_name))
                    pred = cv2.imread(os.path.join(pred_data_dir, args.test_model_name, class_name, img_name), 0)
                    gt = cv2.imread(os.path.join(label_data_dir, class_name, img_name[:-4] + '.png'), 0)
                    for _, fun in mertic_fun.items():
                        fun.step(pred=pred / 255, gt=gt / 255)

            fm = mertic_fun['FM'].get_results()[0]['fm']
            wfm = mertic_fun['WFM'].get_results()['wfm']
            sm = mertic_fun['SM'].get_results()['sm']
            em = mertic_fun['EM'].get_results()['em']
            mae = mertic_fun['MAE'].get_results()['mae']
            fnr = mertic_fun['FM'].get_results()[1]

            eval_res = '{}: Smeasure:{:.4f} || meanEm:{:.4f} || adpEm:{:.4f} || maxEm:{:.4f} || wFmeasure:{:.4f} || ' \
                       'adpFm:{:.4f} || meanFm:{:.4f} || maxFm:{:.4f} ||  MAE:{:.4f} || fnr:{:.4f}'.format(
                args.test_model_name, sm, em['curve'].mean(), em['adp'], em['curve'].max(), wfm, fm['adp'],
                fm['curve'].mean(), fm['curve'].max(), mae, fnr)
            print(eval_res+"\n")
            log_file.write(eval_res + '\n')
            log_file.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser("CoSOD testing script", parents=[get_args_parser()])
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    cudnn.benchmark = True
    main(args)
    pass

