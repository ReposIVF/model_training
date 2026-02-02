import supervision as sv

sv.DetectionDataset.from_coco(
    images_directory_path='./datasets_tail/AS 211121 Jan 31_Boston_IVF_3101221249_tail-added/images',
    annotations_path='./datasets_tail/AS 211121 Jan 31_Boston_IVF_3101221249_tail-added/instances_default.json',
    force_masks=True
).as_yolo(
    images_directory_path='./datasets_tail_yolo/AS 211121 Jan 31_Boston_IVF_3101221249_tail-added/images',
    annotations_directory_path='./datasets_tail_yolo/AS 211121 Jan 31_Boston_IVF_3101221249_tail-added/labels_yolo/'
)