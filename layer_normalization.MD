# Layer Normalization

this type of normalization was proposed in [1] and is similar to batch normalization, but instead of normalizing across the batch dimension, it normalizes across the feature dimension. This is useful for recurrent neural networks as it does not require batch size to be fixed. The input shape is (batch_size, num_features) and the output shape is the same as the input shape.

[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. “Layer normalization.” arXiv preprint arXiv:1607.06450 (2016).# Layer Normalization in Deep Learning

Layer Normalization has become a crucial component in many deep learning architectures, particularly in Transformer models. It differs from Batch Normalization in several ways and offers unique advantages, especially in the context of Transformer architectures.

## What is Layer Normalization?

Layer Normalization is a technique to normalize the inputs across the features instead of across the batch dimension as in Batch Normalization. It was introduced by Ba et al. in their paper ['Layer Normalization'](https://arxiv.org/abs/1607.06450) (2016). This normalization technique is particularly effective in stabilizing the hidden state dynamics in recurrent networks.

## Layer Normalization vs Batch Normalization

### Batch Normalization
- Normalizes across the batch dimension.
- Dependence on the batch size can be problematic for small batches.
- Commonly used in Convolutional Neural Networks (CNNs).

### Layer Normalization
- Normalizes across the features.
- Independent of batch size, making it effective for recurrent architectures and online learning.
- Particularly beneficial in Transformer models.

## Why is Layer Normalization Effective in Transformers?

Transformers, unlike CNNs, do not inherently possess any regularization mechanism such as pooling layers. Layer Normalization helps in stabilizing the training of Transformers by normalizing the inputs across the features. It is crucial in managing the internal covariate shift and ensuring that the scale of activations remains consistent across different layers and inputs.

- **Reference**: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv preprint arXiv:1607.06450. [Link to paper](https://arxiv.org/abs/1607.06450)

## Applications in Transformers

- Stabilizes the training process.
- Improves convergence speed.
- Reduces sensitivity to hyperparameter settings.

it is interesting to note this paper: **Leveraging Batch Normalization for Vision Transformers** [Link to paper](https://arxiv.org/abs/2102.08602) which shows that Batch Normalization can be used in Vision Transformers to achieve better performance than Layer Normalization.




interesting paper:

https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/papers/Yao_Leveraging_Batch_Normalization_for_Vision_Transformers_ICCVW_2021_paper.pdf