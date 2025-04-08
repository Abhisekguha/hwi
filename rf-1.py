import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(history)

plt.figure(figsize=(12, 8))

plt.plot(
    df['epoch'],
    df['train_loss'],
    label='Training Loss',
    marker='o',
    linestyle='--'
)

plt.plot(
    df['epoch'],
    df['test_loss'],
    label='Validation Loss',
    marker='o',
    linestyle='--'
)

plt.title('Train/Validation Loss over Epochs')
plt.xlabel('Epoch')






import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(history)

df['avg_precision'] = df['test_coco_eval_bbox'].apply(lambda arr: arr[0])
df['avg_recall'] = df['test_coco_eval_bbox'].apply(lambda arr: arr[6])

plt.figure(figsize=(12, 8))
plt.plot(
    df['epoch'],
    df['avg_precision'],
    marker='o',
    linestyle='--'
)
plt.title('AP (IoU=0.50:0.95, area=all, maxDets=100) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AP')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(
    df['epoch'],
    df['avg_recall'],
    marker='o',
    linestyle='--'
)
# The rest of the plotting code for 'avg_recall' seems to be cut off in the image.
# Assuming it would have similar title, labels, grid, and show commands:
plt.title('AR (IoU=0.50:0.95, area=all, maxDets=100) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AR')
plt.grid(True)
plt.show()



plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()




import supervision as sv

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/test",
    annotations_path=f"{dataset.location}/test/_annotations.coco.json",
)



from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image

# Assuming 'ds' and 'model' are defined elsewhere in your code

# Load the first image and perform prediction
path, image, annotations = ds[0]
image = Image.open(path)
detections = model.predict(image, threshold=0.5)

# Calculate annotation parameters (only need to do this once as image size is the same)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

# Initialize annotators (only need to do this once)
bbox_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_thickness=thickness,
    smart_position=True
)

# Create labels for the ground truth annotations
annotations_labels = [
    f"{ds.classes[class_id]}"
    for class_id
    in annotations.class_id
]

# Create labels for the model's detections
detections_labels = [
    f"{ds.classes[class_id]} (confidence:{confidence:.2f})"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# Annotate the image with ground truth annotations
annotation_image = image.copy()
annotation_image = bbox_annotator.annotate(scene=annotation_image, detections=annotations)
annotation_image = label_annotator.annotate(scene=annotation_image, detections=annotations, labels=annotations_labels)

# Annotate the image with the model's detections
detections_image = image.copy()
detections_image = bbox_annotator.annotate(scene=detections_image, detections=detections)
detections_image = label_annotator.annotate(scene=detections_image, detections=detections, labels=detections_labels)

# Plot the annotated images
sv.plot_images_grid(images=[annotation_image, detections_image], grid_size=(1, 2), titles=["Annotation", "Detection"])


