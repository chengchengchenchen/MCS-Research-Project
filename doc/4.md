## **Slide 1 — Title Page**

Good afternoon everyone. My name is Jingcheng Qian, and today I’ll be presenting my research on *“Data Quality of Generative Images for Object Detection.”*

------

## **Slide 2 — Introduction**

**Example:**

In the past decade, object detection has become powerful and easy to use.
 Thanks to open-source models and affordable hardware, almost anyone can train a detector — even on a laptop.

But what really limits progress today is still **data**, not the model.
 In academia, we usually focus on improving architectures using large, public datasets.
 But in the **industry**, it’s a very different story — companies often need to detect **uncommon or new types of objects**, and they usually have **very limited labeled data** to work with.

That’s where **Generative AI** comes in.
 Diffusion models can now create **realistic and diverse AI-generated scenes**, offering a way to **expand datasets** when real samples are limited.

*(In the figure here — the top shows a real photo captured at dusk, and the bottom is a diffusion-generated version — similar scene, but created entirely from text.)*

However, the quality and usefulness of such data for detection are still uncertain.

So, in collaboration with **Swordfish Computing**, this research explores how generated data improve accuracy and under what conditions.



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

**ControlNet** plays a key role by adding structure — like edges, depth maps, or segmentation layouts — so that the generated images maintain accurate spatial relationships and respect the original layout.

**Low-Rank Adaptation**, Originally developed for large language models, LoRA was designed to fine-tune massive networks efficiently by updating only a small number of parameters instead of retraining the entire model. This idea has since been successfully transferred to diffusion models, allowing them to adapt quickly to new visual domains. For instance, converting daytime scenes into foggy or nighttime environments — without full retraining.

Together, these components make diffusion models both controllable and adaptable.

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

Training proceeds with different ratios of real and generated data, then we evaluate both the visual quality and the detection performance. The feedback from each stage is used to refine the generation process, so that improvements are measurable and repeatable.

------

## **Slide 9 — Dataset**

For the dataset setup, we follow a staged strategy.

We begin with benchmark datasets like COCO and PASCAL VOC to validate the pipeline. 

To simulate real-world data shortage, we also use reduced subsets with only **a few dozen images per class**.

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

Next, **perceptual quality** is assessed using *FID* and *IS* scores to evaluate the fidelity and diversity of the generated images. Lower FID and higher IS values indicate more realistic and varied results.

To measure **semantic consistency**, I use *CLIP* and *DINOv2* similarity scores to capture how well an image’s content aligns with its intended labels or descriptions. Specifically, **CLIP** measures image–text alignment, while **DINOv2** captures semantic similarity between visual features.

*(Here on the slide, you can see examples of generated images with different semantic scores. Images with higher  scores appear more faithful to the intended object.)*

Finally, **human evaluation** complements these automated metrics. Human reviewers  rate the realism and annotation accuracy of generated samples, providing human insights that just numerical scores may miss.

Together, these layers of evaluation reveal how visual quality relates to real detection performance, offering a comprehensive understanding of data usefulness.



## **Appendix:** **Diffusion Model**

A diffusion model gradually transforms random noise into a realistic image through a learned denoising process, essentially “drawing” an image step by step.



###  **1. Why do we need generative images instead of collecting more real data?**

**Answer:** Collecting and labeling real data is expensive and time-consuming. Generative data allows rapid and low-cost expansion of datasets, especially for rare or safety-critical scenarios.

------

###  **2. What is a diffusion model in simple terms?**

**Answer:** A diffusion model gradually transforms random noise into a realistic image through a learned denoising process, essentially “drawing” an image step by step.

------

###  **3. How realistic are these generated images compared to real ones?**

**Answer:** Many generated images are visually indistinguishable from real photos, especially after fine-tuning with LoRA and structural control using ControlNet.

------

###  **4. How is this research useful for industry applications?**

**Answer:** It helps companies improve object detection when they have limited real data, reducing data collection costs and enabling faster model deployment.

------

###  **5. Among all the metrics, which one is the most important?**

**Answer:** For object detection, *mean Average Precision (mAP)* is the most critical, as it directly measures how well the model detects objects in real-world tasks.

------

###  **6. Can training on generated data harm real-world performance?**

**Answer:** If used carefully, no — but overusing synthetic data can reduce realism. The study explores optimal mixing ratios to balance diversity and authenticity.

------

###  **7. How do ControlNet and LoRA actually help in generation?**

**Answer:** ControlNet ensures structural consistency (like object layout), while LoRA adapts the diffusion model to specific domains, improving realism and relevance.

------

###  **8. How do you ensure the generated images have correct labels?**

**Answer:** The pipeline reuses original bounding boxes and semantic information, so generated images remain aligned with existing annotations automatically.

------

###  **9. What kinds of objects or scenarios benefit most from generative data?**

**Answer:** Rare or difficult conditions — such as fog, night, or emergency scenes — benefit the most because they are hard to capture in large quantities.

------

###  **10. How do you measure the “usefulness” of generated data ?**

**Answer:** By combining visual quality metrics (like FID and CLIP scores) with downstream detection accuracy (mAP), we can see how visual realism translates into actual task improvement.

###  **11. The meaning of human evaluation?**

Human evaluation helps us check what automated metrics can’t capture — mainly how realistic and correctly labeled the generated images are.
 It provides a human perspective on visual quality and consistency.
 This way, we can confirm whether improvements shown by numbers actually make sense in practice.
