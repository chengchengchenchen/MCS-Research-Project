# Agenda 12

## Summary

1. prompt construction
2. CLIP computation: from computing on the entire image to computing within each bounding box
3. read some papers about using inpainting for data augmentation

## **Roadmap**

1. Apply the modifications in the generation module to the pipeline and obtain the results.   **next week**
2. Try using inpainting for background reconstruction as a form of data augmentation. **after travel**



### Prompt constrution

extract all object category labels from the annotations and concatenating them， like

`a scooter, a scooter, a scooter`
`a picture of scooter, a picture of scooter, a picture of scooter`



### CLIP computation

During preliminary experiments, I observed that using a global CLIP score to filter synthetic images did not work well (nearly equivalent to random filtering).

By reading the source code and the Method section of *Data Augmentation for Object Detection via Controllable Diffusion Models* [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.pdf), I find it computes CLIP scores *per bounding box*.

I  re-implemented the filtering stage with box-level CLIP scoring (as in the paper), while retaining DINOv2 global feature computation (from *Scaling Inference Time Compute for Diffusion Models* [CVPR 2025]([Scaling Inference Time Compute for Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.pdf))), and will re-evaluate its effectiveness in the pipeline next week.



### Inpainting for Data augmentation

1. Use diffusion-based inpainting to modify specific regions of the original image for:

   1. **Object augmentation**

   2. **Background augmentation**

2. Reuse the original mask/bounding-box annotations.

The paper *A Simple Background Augmentation Method for Object Detection with Diffusion Models* [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08374.pdf) finds that **background augmentation is more effective than object augmentation**.

![](..\assets\background_aug.png)

Can try using a state-of-the-art inpainting model such as FLUX with this method as the generation component of the pipeline.



### Travel

Travel from Melbourne to Adelaide on **Nov 30**, and return to Melbourne on **Dec 6**.
Estimated costs: **~$400 for flights** and **~$2000 for 6 nights of accommodation**, based on preliminary searches on Booking.com.

Based on the email, will visit to Swordfish Computing in the afternoon Dec 2. (not sure)





1. I updated the generation module code last week, and this week I will run the full pipeline. 
2. Whether it’s possible to take a bus from Adelaide Airport to the city?
3. Is Louis still unsure about attending? Should I go ahead and book my own flights and accommodation?
4. If I stay in Adelaide for the weekend and book a flight back on Sunday, will that affect the reimbursement?

