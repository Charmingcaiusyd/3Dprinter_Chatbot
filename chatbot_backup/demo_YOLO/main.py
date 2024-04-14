import os
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def main(image_path):
    # 设置模型文件的相对路径
    model_folder = "models/"
    model_path_m = os.path.join(model_folder, "yolov8m.pt")
    model_path_n = os.path.join(model_folder, "yolov8n.pt")

    # 加载模型
    model_m = YOLO(model_path_m)
    model_n = YOLO(model_path_n)

    # 从图像文件路径中创建一个PIL图像对象
    img = Image.open(image_path).convert('RGB')

    # 进行预测
    results_m = model_m([img])
    results_n = model_n([img])

    # 从列表中获取Results对象
    results_m_obj = results_m[0]
    results_n_obj = results_n[0]

    # 获取边界框信息
    boxes_m = results_m_obj.boxes.xyxy
    boxes_n = results_n_obj.boxes.xyxy

    # 假设模型M检测的是类别0和1，模型N检测的是类别2和3
    labels_m = torch.tensor([0, 1] * len(boxes_m))  # 创建一个标签张量，包含模型M的检测结果的标签
    labels_n = torch.tensor([2, 3] * len(boxes_n))  # 创建一个标签张量，包含模型N的检测结果的标签

    # 合并两个模型的结果
    all_boxes = torch.cat((boxes_m, boxes_n), dim=0)  # 按行连接两个张量
    all_labels = torch.cat((labels_m, labels_n), dim=0)  # 按行连接两个张量

    # 创建一个新的图像用于显示结果
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    # 循环遍历每个检测结果，并在图像上绘制边界框和标签
    for box, label in zip(all_boxes, all_labels):
        x_min, y_min, x_max, y_max = box[:4]
        if label in [0, 1, 2, 3, 4]:  # 检查标签是否是你关心的
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min, f'Class {label}', color='red')


    # 保存结果图像到文件
    result_image_path = 'result_image.jpg'
    plt.savefig(result_image_path)
    plt.close(fig)  # 关闭图形窗口

    # 返回新图像的路径，如果需要的话，可以通过Django返回给前端
    response_data = {
        "result_image_path": result_image_path,
    }
    print(response_data)


if __name__ == "__main__":
    image_path = 'image.jpg'
    main(image_path)

