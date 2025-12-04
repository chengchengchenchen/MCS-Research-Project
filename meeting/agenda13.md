# Agenda 13

## Summary

1. Refine the metrics  for evaluating the quality of generated images and filtering methods in pipeline.

## **Roadmap**

1. Conduct more grid experiments and collect results. **(Next week)**
2. Experiment with adding a LoRA module. **(Next week)**
3. Explore using inpainting for background reconstruction as a form of data augmentation. **(After the trip)**



### Metrics

Proposed metrics for assessing generated image quality:

1. DINO score between the synthesized image and the source image.
    Measures structural and appearance-level similarity based on self-supervised visual features.

2. CLIP score between the synthesized image and the source image.
   Measures high-level semantic consistency between the generated scene and the original scene.

   *Scaling Inference Time Compute for Diffusion Models [(CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html)*

3. CLIP score between each bounding box in the synthesized image and its corresponding prompt.
    Evaluates whether each object visually aligns with its category description.

    *Data Augmentation for Object Detection via Controllable Diffusion Models [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.pdf)* 



### Filtering and Ranking

These three metrics exhibit a certain degree of positive correlation.

![realtive](..\assets\correlations.png)

In the original filtering strategy, for each source image we simply selected the top-k synthetic images. However, for source images with **simple structure**, most generated results tend to be of reasonably good quality, whereas for source images with **complex structure**, nearly all generated results are of relatively poor quality.

To make full use of the three metrics to effectively rank the quality of generated images, in order to select high-quality synthetic samples to mix with original data for training and improve object detection performance.

1. Normalize all metrics.
2. Filter out low-quality synthesized samples.
3. Create a composite ranking method for synthesized images.



### Few shot

![](..\assets\map_bar.png)

### Prompt Design

Although *Data Augmentation for Object Detection via Controllable Diffusion Models* [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.pdf) only mentions using classification labels to construct prompts, experiments show that adding realistic style descriptors such as `real world` and `camera shooting` improves mAP when training with mixed data.

Further experiments will be conducted after introducing a real world style LoRA module.

