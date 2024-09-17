from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2

# Load COCO annotations from the JSON file
coco = COCO('merged_coco.json')

# Get all image IDs in the dataset
img_ids = coco.getImgIds()[::-1]

# Loop through the image IDs
for img_id in img_ids:
    # Load image metadata
    img = coco.loadImgs(img_id)[0]

    # Load the image from file
    image_path = f"/Users/abdul/Desktop/Dataset/dataset/{img['file_name']}"
    image = plt.imread(image_path)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Show the annotations (bounding boxes, segmentation, etc.)
    coco.showAnns(anns)
    
    # Display the image with annotations
    plt.title(f"Image: {img['file_name']}")
    plt.show()

    # Print the annotations below the image
    for ann in anns:
        print(f"Annotation ID: {ann['id']}")
        print(f"Category ID: {ann['category_id']}")
        print(f"Bounding Box: {ann['bbox']}")
        print(f"Segmentation: {ann['segmentation']}")
        print(f"Area: {ann['area']}")
        print(f"Is Crowd: {ann['iscrowd']}")
        print("\n" + "="*30 + "\n")
