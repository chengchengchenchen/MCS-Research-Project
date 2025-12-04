## **Slide 1 — Title Page**

Good afternoon everyone. My name is Jingcheng Qian, and today I’ll be presenting my research on *“Data Quality of Generative Images for Object Detection.”*

------

## **Slide 2 — Introduction**

**Example:**

Modern object detection is powerful, but it is extremely **data-hungry** and depends on large labeled datasets.
In many real-world applications, we only have **limited data**, especially for rare or new object types, and collecting more is costly and slow.

**Generative AI**, especially diffusion models, offers a way to create realistic, diverse synthetic scenes to expand these small datasets.

*(In the figure here — the top shows a real photo captured at dusk, and the bottom is a diffusion-generated version — similar scene, but created entirely from text.)*

However, the quality and usefulness of such data for detection are still uncertain. In this project with Swordfish Computing, we study when and how such AI-generated data actually improve detection accuracy.



## **Slide 3 — Outline**

So, here’s the outline for today’s talk.

I’ll start with some **background** — what’s the key **data challenge** and why it matters.
 Then I’ll briefly go over **current methods** and their limitations.

After that, I’ll introduce **controllable diffusion models** and discuss the **main research questions** and our **objectives**.

Finally, I’ll explain the overall framework, including the **methodology**, **dataset**, **generation pipeline**, and how we **evaluate** the results.

------

## **Slide 4 — Data Bottleneck**

Modern object detection systems are extremely data-hungry — they require massive amounts of well-labeled images.
 But in practice, collecting and annotating diverse real-world scenes is both **expensive and time-consuming**.
 Even after all that effort, datasets are often unbalanced.
 Common situations, like clear daytime driving, appear in abundance, while rare but safety-critical ones — such as fog, night, or emergencies — are scarce.
 This imbalance, known as the **long-tail problem**, creates a major data bottleneck.
 As a result, models perform well on frequent cases but struggle to generalize to rare or unseen scenarios.

------

## **Slide 5 — Traditional Methods**

Researchers have explored several methods to overcome the data bottleneck.*(These three figures show examples of the methods)*
 First, **traditional augmentation** techniques like rotation, cropping, or pixel mixing can slightly improve diversity, but they only modify existing images and can’t create new or unseen scenes.
 Second, **simulation-based datasets**, like GTA-V, automatically generate labeled scenes at scale. However, these images often appear too clean or artificial, lacking the natural lighting and textures of real photos.
 Finally, **domain adaptation** techniques try to make generated images look more realistic by transferring styles, but some visual inconsistencies always remain.

Despite these efforts, there’s still a clear gap between generated and real data. This is where modern generative models start to make a difference.

------

## **Slide 6 — Controllable Diffusion Models**

Recent advances in **diffusion models** have opened a new path for data generation.
 Unlike traditional augmentation, these models can actually *create* new and realistic scenes, instead of simply transforming existing pixels.

Here are 2 awesome tools to help generate controllable and adaptable images

**ControlNet** plays a key role by adding structure — like edges, depth maps, or segmentation layouts — so that the generated images maintain accurate spatial relationships and respect the original layout.

**Low-Rank Adaptation**,  designed to fine-tune massive networks efficiently by updating only a small number of parameters instead of retraining the entire model, allowing them to adapt quickly to new visual domains. For instance, converting daytime scenes into foggy or nighttime environments — without full retraining.



------

## **Slide 7 — Research Objectives **

**Building on these ideas, this research aims to evaluate the effectiveness and reliability of generated data in object detection.**
 To achieve this goal, the study focuses on several key tasks:
**quantify the domain gap** between real and generated data using multiple visual and semantic metrics.
**evaluate cross-domain robustness**, examining how well models trained with generative data perform when applied to real-world images.
**assess the realism and annotation consistency** of generated samples, ensuring they are both visually convincing and semantically accurate.
**develop best practices** for using generative data effectively in object detection training pipelines.

And additionally, **Identify the contribution of individual components** in the generative augmentation pipeline through **ablation studies**.

------

## **Slide 8 — Research Framework**

The research follows a four-stage framework: **generation**, **filtering**, **training**, and **evaluation**.**(**Figure 6 shows the pipeline of our closed-loop research framework)

In the generation stage, diffusion models enhanced with ControlNet and LoRA are used to produce controllable and adaptive generated data. The filtering stage then selects high-quality samples using visual and semantic metrics.

Training proceeds with different ratios of real and generated data, then we evaluate both the visual quality and the detection performance.

------

## **Slide 9 — Dataset**

For the dataset setup, we follow a staged strategy.

We begin with benchmark datasets like COCO and PASCAL VOC to validate the pipeline. 

To simulate real-world data shortage, we also use reduced subsets with only **a few images per class**.

Then, we move to **niche datasets** from domains like self-driving, focusing on rare objects and difficult conditions.

We also fine-tune the diffusion model using **LoRA** on small real-world image sets to better capture realistic lighting and textures.

This setup lets us compare performance under both **rich** and **limited** data conditions, showing how well the generative augmentation adapts to different domains.

------
## **Slide 10 — Generation Pipeline**

The generation pipeline is developed using ComfyUI, a modular and node-based workflow system that’s flexible and easy to reproduce.

Using a generation example, we start with an **original image**, which is first converted into a **Canny edge map**.
That edge map is then fed into **ControlNet**, guiding the model to make a new image with the same layout.

Next, **LoRA** helps adapt the model to different domains.

Finally, we reuse the original bounding-box annotations and apply them directly.
This pipeline automatically produces realistic, well-labeled images at scale — greatly reducing manual labeling.

------
## **Slide 11 — Training **

After generating the generated dataset, we move to the downstream task.
 Here, I focus on some representative object detectors: **YOLO** and **others**. These models balance accuracy and real-time efficiency, making them suitable for practical applications like self-driving.

The **training strategy** involves mixing real and generated data in different ratios — from entirely real to heavily generated — to explore how generative data contributes under various levels of limited data.

The main **objective** of this stage is to quantify the value of generated data and identify the most cost-effective balance between real and generated samples. This helps determine how much real data can be replaced without significant performance loss.

## Slide 12 — Evaluation

After training, the models are evaluated through multiple perspectives to understand both task performance and data quality.

For **detection performance**, I use standard object detection metrics such as *mean Average Precision (mAP)* and *AP50*, which directly measure how well the model finds and locates objects. These are task-oriented indicators that reflect real-world performance.

To measure **semantic consistency**, I use *CLIP* and *DINOv2* similarity scores to capture how well an image’s content aligns with its intended labels or descriptions. Specifically, **CLIP** measures image–text alignment, while **DINOv2** captures semantic similarity between visual features.

*(Here on the slide, you can see examples of generated images with different semantic scores. Images with higher  scores appear more faithful to the intended object.)*





