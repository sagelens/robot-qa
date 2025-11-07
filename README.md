## Prompt-based Segmentation (Cracks and Drywall taping)

### Task

The task requires fine tuning a text-conditioned segmentation model which outputs binary mask

### Inputs

MODEL (IMAGE + User Prompt) --> Bounding box + Segmentation mask (0, 255)

### Datasets
- Dataset 1 (Taping area):
https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

Prompt Prefix: “segment drywall”

- Dataset 2 (Cracks): https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36
Prompt prefix: “segment cracks”

### Model used and Fine Tuned

- `google/paligemma-3b-mix-224`

### Approach

- Both datasets provide raw RGB images (640, 640) and detection bounding boxes.
- Exported images and annotations from RoboFlow for both datasets (cracks, drywall) as JSONL files.
- Since dataset has no pre-labelled annotations for segmentation masks (pixel-level). Using the bounding boxes annotations I ran Segment Anything model to obtain high quality pixel masks for each given RGB image and bounding box (this uses src/convert_cracks.py and src/convert_drywall.py)
- After obtaining pixel maps I saved them as ground truth segmentation masks (pseudo labels) for each image in train and valid splits.
- For each object, I generate location tokens by normalizing its bounding box coordinates relative to the image size, then binning these 0-to-1 values into 1024 discrete integer IDs, which are mapped to four <loc> tokens.

Simultaneously, I generate segmentation tokens by cropping the mask to the bounding box, resizing this crop to a fixed 64x64, and using a pre-trained encoder ("vae-oid.npz") to convert this visual patch into a sequence of <seg> tokens.

These two token strings are then concatenated together, followed by the object's class name (e.g., <loc####><seg####> cracks).

If there are multiple objects, each object's full string is joined by a " ; " separator to create the final, complete output string.


### Results

#### Train vs Validation Metrics

| Segment Type | Metric Type  | Train mIoU | Train mDice | Valid mIoU | Valid mDice |
| ------------ | ------------ | ---------- | ----------- | ---------- | ----------- |
| Cracks       | Segmentation | 0.2381     | 0.3338      | 0.2745     | 0.3723      |
| Cracks       | Bounding Box | 0.6139     | 0.6935      | 0.5998     | 0.6688      |
| Drywall      | Segmentation | 0.0905     | 0.1486      | 0.0916     | 0.1541      |
| Drywall      | Bounding Box | 0.1462     | 0.2353      | 0.1525     | 0.2471      |

#### mIoU consistency statistics

| Statistic          | Train           | Validation      |
| ------------------ | --------------- | --------------- |
| Mean               | 0.2180          | 0.1826          |
| Std Dev            | 0.2291          | 0.2171          |
| Min-Max            | 0.0000 – 0.9938 | 0.0000 – 0.9929 |
| P10                | 0.0103          | 0.0091          |
| P50                | 0.1336          | 0.0971          |
| P90                | 0.5632          | 0.5293          |
| Cracks mIoU ± Std  | 0.2381 ± 0.2356 | 0.2745 ± 0.2600 |
| Drywall mIoU ± Std | 0.0905 ± 0.1198 | 0.0916 ± 0.1011 |
---

### **Segmentation Visuals**

Sample qualitative results visualizing segmentation outputs from the model.
Each image corresponds to either **crack** or **drywall** segmentation.

#### **Cracks Segmentation**

![Cracks Example 1](visuals/3872_jpg.rf.f4ec798bfc519354376209d0a8d6e3b1_cracks_canvas.jpg)
![Cracks Example 2](visuals/cracking0102204_jpg.rf.0d64df9e69ef678b6185988b7e030387_cracks_canvas.jpg)
![Cracks Example 3](visuals/WP6S2G7E75FFPFLA3RLQBAQGWQ_jpg.rf.59e85935f19d5885a4e5dd981eb243fe_cracks_canvas.jpg)

#### **Drywall Segmentation**

![Drywall Example 1](visuals/2000x1500_5_resized_jpg.rf.6423c25a172441fa1f66e4e927fe68a9_drywall_canvas.jpg)
![Drywall Example 2](visuals/IMG_8209_JPG_jpg.rf.8fbaa3c995fac30818b643910225b928_drywall_canvas.jpg)
![Drywall Example 3](visuals/IMG_8223_JPG_jpg.rf.121531b6267b5a04f871202c0dbc43c6_drywall_canvas.jpg)
