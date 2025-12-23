# significance_test.py
import numpy as np
from scipy import stats

def load_iou(file_path):
    """
    读取 IOU 文件，每行格式：image_name iou_value
    返回 numpy 数组（只保留 IOU 值）
    """
    iou_list = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                iou_list.append(float(parts[1]))
    return np.array(iou_list)

def main():
    # 替换为你保存的 IOU 文件
    baseline_file = 'IOU/BPV.txt.txt'
    my_file = 'IOU/HPV.txt.txt'

    baseline_iou = load_iou(baseline_file)
    my_iou = load_iou(my_file)

    # 检查长度是否一致
    if len(baseline_iou) != len(my_iou):
        raise ValueError("两组 IOU 样本数量不一致")

    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(my_iou, baseline_iou)
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

    # 置信区间计算
    diff = my_iou - baseline_iou
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    ci95 = 1.96 * std_diff / np.sqrt(n)  # 95% CI
    print(f"Mean IOU improvement: {mean_diff:.4f} ± {ci95:.4f} (95% CI)")

    # 可选：保存差值到文件
    with open('IOU_diff.txt', 'w') as f:
        for i, d in enumerate(diff):
            f.write(f"{i} {d:.6f}\n")
    print("差值已保存到 IOU_diff.txt")

if __name__ == "__main__":
    main()
