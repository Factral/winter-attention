The uses of Transformer architectures has become a common topic in computer vision related tasks, this type of models has demonstrated a superior performance in task such as image classification, object detection, image segmentation or action recognition.

The transformer architecture was introduced to computer vision in the article ['An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale'](https://arxiv.org/pdf/2010.11929v2.pdf) by Dosovitskiy et al. (2020).

## How vision transformer works?

Let's dive deep into the ViT (Vision Transformer) and how it works!!

In ViT the images are represented as sequences like in NLP tasks, so the **first step in processing images with ViT is split images into fixed-size patches**, 

![[patchify.png]]

Then each patch is flattened into a single vector, this allows ViT to treat patches as sequences of data. Then with learnable linear projection these flattened patches are mapped into a lower-dimensional space.

But now we face a problem, how to preserve information order during the process? Keep in mind that this order is not inherent and the transformer itself can't capture it. This is where trainable positional embedding comes into play, these embeddings help the model to retain information about spatial relationships between patches.

Before feed the sequence of patches to transformer encoder and perform classification, a learnable class token is added, for each patch sequence it's initialize as zeros. Use this "blank state" token pushes the model to learn a general representation of the image (patch sequence).

Now, finally, the patch sequence is feed into the transformer block for pre-training.
## Attention visualization
## Transfer learning (Fine-tuning)
## Vision Transformers vs Convolutional Neural Networks

## Advantages and disadvantages of Vision Transformers

