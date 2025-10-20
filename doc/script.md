## **Slide 1 – Title Page (0:00–0:30)**

> **Good afternoon everyone.**
>  My name is *Jingcheng Qian*, and today I’m presenting my research proposal titled **“Data Quality of Generative Images for Object Detection.”**
>
> In this presentation, I’ll first introduce the **problem and motivation**, then describe the **research framework and methodology**, followed by the **evaluation strategy**, and finally conclude with the **expected contributions**.

------

## 🕐 **Slide 2 – Background: Data Bottleneck (0:30–2:00)**

> Let’s begin with the background and the problem we face.
>
> Object detection models are **data-hungry** — they require large volumes of **precisely annotated images**, often with bounding boxes or masks.
>
> However, creating such annotations is **time-consuming and costly**.
>  Traditional data augmentation methods—like flipping or rotating—can only transform **existing pixels**; they cannot create **new scenes** or **rare events**.
>
> *(Show Figure 1: examples of traditional augmentation vs unseen rare event)*
>
> This limitation is especially severe in **autonomous driving**, where rare but safety-critical scenes—such as accidents at night or heavy fog—are hard to capture.
>
> In short, we face a **data bottleneck**: detectors crave more data, but labeling is expensive and limited.

------

## 🕐 **Slide 3 – Motivation & Research Objectives (2:00–3:00)**

> Recent advances in **generative AI**—especially **diffusion models**—provide a new opportunity.
>
> These models can produce **high-fidelity** and **controllable synthetic images**.
>  But even with realistic appearance, synthetic data still shows a **domain gap** from real data—differences in texture, lighting, and semantics.
>
> **ControlNet** and similar architectures improve spatial alignment, but their impact on detection performance hasn’t been systematically studied.
>
> This leads to the **core research question:**
>
> > *Can diffusion-generated synthetic images effectively supplement or even replace real annotated data for object detection?*
>
> To answer this, my research sets four key objectives:
>
> - Quantify the **domain gap** between real and synthetic data.
> - Evaluate **cross-domain robustness** of detectors trained on synthetic data.
> - Assess the **quality** of generated images in realism and annotation consistency.
> - Provide **practical guidelines** for how to best integrate synthetic data into training.

------

## 🕐 **Slide 4 – Research Framework (3:00–4:00)**

> To achieve these objectives, I propose a **closed-loop research framework**.
>
> *(Show Figure 3: the pipeline diagram)*
>
> It consists of four main stages:
>
> 1. **Controlled data generation**,
> 2. **Quality filtering**,
> 3. **Model training**, and
> 4. **Multi-level evaluation.**
>
> This pipeline enables a systematic investigation — from generation all the way to detection — allowing us to measure the **true effectiveness of generative data**.

------

## 🕐 **Slide 5 – Methodology: Generation Pipeline (4:00–6:00)**

> Let’s start with the **generation stage.**
>
> The pipeline will be built on **ComfyUI**, a modular node-based framework that allows visual control and reproducibility.
>
> The **core generative models** include **Stable Diffusion** and **FLUX**, which represent state-of-the-art diffusion backbones for image synthesis.
>
> To ensure that generated objects appear in the **right location**, I’ll use **ControlNet**, which takes visual priors as input and constrains the image structure during denoising.
>  *(Show Figure 2: ControlNet structure)*
>
> To further improve realism, I’ll apply **LoRA (Low-Rank Adaptation)** for lightweight domain fine-tuning, aligning textures and lighting to real-world datasets.
>
> Together, these modules form a flexible, controllable, and efficient **synthetic data generation pipeline**.

------

## 🕐 **Slide 6 – Methodology: Downstream Task (6:00–7:00)**

> Next comes the **downstream task**: object detection.
>
> I’ll use two representative detectors — **YOLO** for real-time inference, and **RT-DETR** for transformer-based precision.
>
> The key idea is to vary the **ratio of real to synthetic data** during training.
>  *(Show Table 1: planned ratios)*
>
> For example, models will be trained with 100%, 50%, or even 0% real data.
>  This helps quantify **how much synthetic data** can replace real annotations while maintaining performance.
>
> The experiments will provide direct evidence of the *trade-off between cost and accuracy.*

------

## 🕐 **Slide 7 – Evaluation Framework (7:00–8:30)**

> The evaluation covers both **task-level performance** and **data-level quality.**
>
> **At the task level**, I’ll measure:
>
> - *mAP* and *AP50* to assess detection accuracy.
>
> **At the data level**, I’ll use a set of complementary metrics:
>
> - *FID* and *Inception Score* for fidelity and diversity,
> - *CLIP* and *DINOv2* scores to check semantic and structural consistency,
> - and **human evaluation**, where participants rate realism and label faithfulness.
>
> *(Show Figure 5: correlation between CLIP/DINOv2 and mAP)*
>
> By analyzing how these scores correlate with mAP, we can understand **what kind of generative data is truly useful** for training detectors.

------

## 🕐 **Slide 8 – Expected Contributions (8:30–10:30)**

> Based on this framework, the research is expected to produce **both theoretical and practical contributions.**
>
> **First**, it will **quantify the domain gap** between real and synthetic data, explaining how appearance differences affect detection accuracy.
>
> **Second**, it will **demonstrate the transferability** of detectors trained on synthetic data, showing when and how they generalize to real-world images.
>
> **Third**, it will **establish evaluation criteria** linking image realism, diversity, and annotation consistency to task performance — a bridge between generative metrics and applied detection.
>
> **Finally**, it will **formulate practical guidelines** for integrating synthetic data into industrial training pipelines — valuable for domains like autonomous driving and robotics.

------

## 🕐 **Slide 9 – Conclusion & Outlook (10:30–12:00)**

> To conclude —
>
> This project tackles a key challenge in modern computer vision:
>  **how to make generative data genuinely valuable for object detection.**
>
> By building a controlled, modular pipeline using **ComfyUI, ControlNet, and LoRA**, and by rigorously evaluating both data quality and detection outcomes,
>  this study aims to provide a clear answer to that question.
>
> The results will deepen our understanding of *data quality in generative AI*, and contribute to more efficient, scalable, and reliable detection models.
>
> Thank you for listening.