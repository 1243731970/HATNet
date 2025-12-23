import os
import argparse
import cv2
import numpy as np
import metric as M


def get_args_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='Evaluating from the predicted images.')
    parser.add_argument('-data_root', type=str, default='../dataset/PV/test/gt')
    parser.add_argument('-pred_root', type=str, default='../results/VGG')
    parser.add_argument('--eval_dataset_list', nargs='+', default=['PV'])
    return parser


def get_metric_function():
    return {
        'FM': M.Fmeasure_and_FNR(),
        'WFM': M.WeightedFmeasure(),
        'SM': M.Smeasure(),
        'EM': M.Emeasure(),
        'MAE': M.MAE()
    }


def main(args):
    model_name = os.path.abspath('').split('/')[-2]
    print(model_name)

    dataset_list = args.eval_dataset_list
    data_root = args.data_root
    pred_root = args.pred_root

    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        print('evaluating on {} dataset.'.format(dataset))

        pred_data_dir = os.path.join(pred_root, dataset)
        label_data_dir = os.path.join(data_root)

        log_file = open('./result/{}.txt'.format(dataset), 'a')

        mertic_fun = get_metric_function()

        print('for model {}.'.format(model_name))

        pred_list = os.listdir(pred_data_dir)
        label_list = os.listdir(label_data_dir)

        for l in range(len(pred_list)):
            img_name = pred_list[l]
            # 假设预测图片和标签图片名称匹配
            pred = cv2.imread(os.path.join(pred_data_dir, img_name), 0)
            gt = cv2.imread(os.path.join(label_data_dir, img_name), 0)
            for _, fun in mertic_fun.items():
                fun.step(pred=pred/255, gt=gt/255)

        # 计算和输出评价指标
        fm = mertic_fun['FM'].get_results()[0]['fm']
        wfm = mertic_fun['WFM'].get_results()['wfm']
        sm = mertic_fun['SM'].get_results()['sm']
        em = mertic_fun['EM'].get_results()['em']
        mae = mertic_fun['MAE'].get_results()['mae']
        fnr = mertic_fun['FM'].get_results()[1]

        eval_res = '{}: Smeasure:{:.4f} || meanEm:{:.4f} || adpEm:{:.4f} || maxEm:{:.4f} || wFmeasure:{:.4f} || ' \
                   'adpFm:{:.4f} || meanFm:{:.4f} || maxFm:{:.4f} ||  MAE:{:.4f} || fnr:{:.4f}'.format(
            model_name, sm, em['curve'].mean(), em['adp'], em['curve'].max(), wfm, fm['adp'],
            fm['curve'].mean(), fm['curve'].max(), mae, fnr)

        log_file.write(eval_res+'\n')
        log_file.close()


if __name__ == '__main__':
    ap = get_args_parser()
    args = ap.parse_args()
    main(args)
