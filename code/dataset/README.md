This folder contains the dataset used to train a YOLO detection model **after the image generation step**.

## Placement & Structure

Place your data in the following layout:

```bash
├─dataset
│  ├─test
│  │  ├─.trash_things_test
│  │  ├─images
│  │  └─labels
│  ├─train
│  │  ├─original
│  │  │  ├─images
│  │  │  └─labels
│  │  └─synth
│  │      ├─109_png.rf.42b6205bf83bd08baf2d693388e76906
│  │      ...
│  └─valid
│      ├─original
│      │  ├─images
│      │  └─labels
│      └─synth
│          ├─102_png.rf.06b2881989c28d0af7e098b419fa9e8e
│          ...
```

