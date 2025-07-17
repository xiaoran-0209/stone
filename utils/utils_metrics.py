# import csv
# import os
# from os.path import join
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
#
#
# def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
#     n, c, h, w = inputs.size()
#     nt, ht, wt, ct = target.size()
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#
#     temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
#     temp_target = target.view(n, -1, ct)
#
#     # --------------------------------------------#
#     #   计算dice系数
#     # --------------------------------------------#
#     temp_inputs = torch.gt(temp_inputs, threhold).float()
#     tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
#     fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
#     fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
#
#     score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     score = torch.mean(score)
#     return score
#
#
# def fast_hist(a, b, n):
#     k = (a >= 0) & (a < n)
#     return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
#
#
# def per_class_iu(hist):
#     return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)
#
#
# def per_class_Recall(hist):
#     """
#     计算每个类的召回率 Recall = TP / (TP + FN)
#     """
#     return np.diag(hist) / np.maximum(hist.sum(1), 1)
# def per_class_Precision(hist):
#     """
#     计算每个类的精确度 Precision = TP / (TP + FP)
#     """
#     precisions = np.diag(hist) / np.maximum(hist.sum(0), 1)
#     return precisions  # 返回一个包含每个类别精确度的数组
# def per_class_PA(hist):
#     """计算每个类的像素准确率 PA = TP / (TP + FN)"""
#     PA = np.diag(hist) / np.maximum(hist.sum(1), 1)  # PA基于真实标签的类别总数
#     return PA  # 返回每个类别的像素准确率数组
#
#
#
# def per_Accuracy(hist):
#     return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)
#
#
# def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
#     print('Num classes', num_classes)
#     hist = np.zeros((num_classes, num_classes))
#
#     gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
#     pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]
#
#     for ind in range(len(gt_imgs)):
#         pred = np.array(Image.open(pred_imgs[ind]))
#         label = np.array(Image.open(gt_imgs[ind]))
#
#         if len(label.flatten()) != len(pred.flatten()):
#             print(
#                 'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
#                     len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
#                     pred_imgs[ind]))
#             continue
#
#         hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
#         if name_classes is not None and ind > 0 and ind % 10 == 0:
#             print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; mRecall-{:0.2f}%; Accuracy-{:0.2f}%'.format(
#                 ind,
#                 len(gt_imgs),
#                 100 * np.nanmean(per_class_iu(hist)),
#                 100 * np.nanmean(per_class_PA(hist)),  # 修改点2：使用新的PA计算
#                 100 * np.nanmean(per_class_Recall(hist)),  # 修改点3：单独显示Recall
#                 100 * per_Accuracy(hist)
#             )
#             )
#
#     IoUs = per_class_iu(hist)
#     PAs = per_class_PA(hist)  # 修改点4：单独计算PA
#     Recalls = per_class_Recall(hist)  # 修改点5：单独计算Recall
#     Precisions = per_class_Precision(hist)
#
#     if name_classes is not None:
#         for ind_class in range(num_classes):
#             print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
#                   + '; PA-' + str(round(PAs[ind_class] * 100, 2)) + '; Recall-' + str(
#                 round(Recalls[ind_class] * 100, 2)) \
#                   + '; Precision-' + str(round(Precisions[ind_class] * 100, 2)))
#
#     print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) \
#           + '; mPA: ' + str(round(np.nanmean(PAs) * 100, 2)) \
#           + '; mRecall: ' + str(round(np.nanmean(Recalls) * 100, 2)) \
#           + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
#     return np.array(hist, int), IoUs, PAs, Recalls, Precisions  # 修改点6：返回新增的PAs和Recalls
#
#
# def adjust_axes(r, t, fig, axes):
#     bb = t.get_window_extent(renderer=r)
#     text_width_inches = bb.width / fig.dpi
#     current_fig_width = fig.get_figwidth()
#     new_fig_width = current_fig_width + text_width_inches
#     propotion = new_fig_width / current_fig_width
#     x_lim = axes.get_xlim()
#     axes.set_xlim([x_lim[0], x_lim[1] * propotion])
#
#
# def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
#     fig = plt.gcf()
#     axes = plt.gca()
#     plt.barh(range(len(values)), values, color='royalblue')
#     plt.title(plot_title, fontsize=tick_font_size + 2)
#     plt.xlabel(x_label, fontsize=tick_font_size)
#     plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
#     r = fig.canvas.get_renderer()
#     for i, val in enumerate(values):
#         str_val = " " + str(val)
#         if val < 1.0:
#             str_val = " {0:.2f}".format(val)
#         t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
#         if i == (len(values) - 1):
#             adjust_axes(r, t, fig, axes)
#
#     fig.tight_layout()
#     fig.savefig(output_path)
#     if plt_show:
#         plt.show()
#     plt.close()
#
#
# def show_results(miou_out_path, hist, IoUs, PAs, Recalls, Precisions, name_classes, tick_font_size=12):
#     draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union",
#                    os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
#     print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))
#
#     draw_plot_func(PAs, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PAs) * 100), "Pixel Accuracy",
#                    os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size,
#                    plt_show=False)  # 修改点7：使用真正的PA数据
#     print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
#
#     draw_plot_func(Recalls, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(Recalls) * 100), "Recall",
#                    os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size,
#                    plt_show=False)  # 修改点8：使用Recall数据
#     print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))
#
#     draw_plot_func(Precisions, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precisions) * 100), "Precision",
#                    os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
#     print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))
#
#     with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer_list = []
#         writer_list.append([' '] + [str(c) for c in name_classes])
#         for i in range(len(hist)):
#             writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
#         writer.writerows(writer_list)
#     print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_Recall(hist):
    """
    计算每个类的召回率 Recall = TP / (TP + FN)
    """
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    """
    计算每个类的精确度 Precision = TP / (TP + FP)
    """
    precisions = np.diag(hist) / np.maximum(hist.sum(0), 1)
    return precisions  # 返回每个类别精确度的数组


def per_class_PA(hist):
    """计算每个类的像素准确率 PA = (TP + TN) / (TP + TN + FP + FN)"""
    TP = np.diag(hist)
    TN = hist[0, 0]
    FP = hist[0, 1]
    FN = hist[1, 0]
    PA = (TP + TN) / np.maximum((TP + TN + FP + FN), 1)  # PA基于真实标签的类别总数
    return PA  # 返回每个类别的像素准确率数组


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; mRecall-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA(hist)),  # 使用新的PA计算
                100 * np.nanmean(per_class_Recall(hist)),  # 单独显示Recall
                100 * per_Accuracy(hist)
            ))

    IoUs = per_class_iu(hist)
    PAs = per_class_PA(hist)  # 使用新的PA计算
    Recalls = per_class_Recall(hist)  # 单独计算Recall
    Precisions = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                  + '; PA-' + str(round(PAs[ind_class] * 100, 2)) + '; Recall-' + str(
                round(Recalls[ind_class] * 100, 2)) \
                  + '; Precision-' + str(round(Precisions[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) \
          + '; mPA: ' + str(round(np.nanmean(PAs) * 100, 2)) \
          + '; mRecall: ' + str(round(np.nanmean(Recalls) * 100, 2)) \
          + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))

    return np.array(hist, int), IoUs, PAs, Recalls, Precisions  # 返回新增的PAs和Recalls


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PAs, Recalls, Precisions, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union",
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PAs, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PAs) * 100), "Pixel Accuracy",
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(Recalls, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(Recalls) * 100), "Recall",
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precisions, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precisions) * 100), "Precision",
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))
    # # ============= 新增：Accuracy 的可视化 =============
    # # 计算全局 Accuracy
    # overall_accuracy = per_Accuracy(hist)
    # # 由于 Accuracy 是全局指标（不是 per-class），我们用一个单元素的列表来统一绘图格式
    # accuracy_values = [overall_accuracy]
    # accuracy_names = ["Overall"]  # 横轴标签（只有一个"Overall"）
    #
    # # 调用绘图函数生成 Accuracy 柱状图
    # draw_plot_func(
    #     accuracy_values,
    #     accuracy_names,
    #     f"Accuracy = {overall_accuracy * 100:.2f}%",  # 标题显示具体数值
    #     "Accuracy",  # X轴标签
    #     os.path.join(miou_out_path, "Accuracy.png"),  # 保存路径
    #     tick_font_size=tick_font_size,
    #     plt_show=False
    # )
    # print("Save Accuracy out to " + os.path.join(miou_out_path, "Accuracy.png"))
    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    #     writer.writerow(["Metric", "Value"])
    #     writer.writerow(["Accuracy", overall_accuracy])
    # print("Save Accuracy text data to " + os.path.join(miou_out_path, "accuracy_results.csv"))
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
