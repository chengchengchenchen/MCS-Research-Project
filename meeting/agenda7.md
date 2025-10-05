# Agenda 7

## summary

1. Develop custom ComfyUI nodes
2. Build automated workflow

## Roadmap

1. presentation



## Custom Nodes

Implemented a custom ComfyUI node that takes a source image and a batch of generated images as inputs, computes their CLIP and DINO scores, and then ranks the generated images accordingly. 

An example of the results is provided in the attachment.



## Automated Workflow 

Use the ComfyUI HTTP API to write a lightweight queuing script. Keep the current workflow unchanged and expose the inputs as parameters.

#### paras:

1. source image path
2. batch size
3. output directory



A short Python script can then iterate over the dataset folder, submit one workflow run per image to ComfyUI, and direct the output directory to a subfolder named after each source image. 



### Presentation

Time: Week 11, Friday, 24 October
Format: 12-minute talk followed by a 3-minute Q&A

I will start preparing the slides this week and incorporate feedback next week to refine the deck.



### Question

1. About the training fee, below is one example of how you might use your allowances (copy from CSIRO file sent to me): 

   **Research Support** such as subscription to journals or databases, **cloud  computing & data storage**, specialised software fees, **research & infrastructure  facilities**, ethics & compliance training, focus groups, field research, academic publication costs, costs for disseminating research findings, editing, formatting &  graphic design services, printing costs, publication fees

   Could this funding be used to cover expenses like renting GPUs or purchasing devices?
