for result in results:
    boxes = result.boxes.xyxy  # 检测框坐标
    confidences = result.boxes.conf  # 置信度
    class_ids = result.boxes.cls  # 类别ID

    # 可视化检测框
    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 显示图片
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
