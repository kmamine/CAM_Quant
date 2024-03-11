# CAM_Quant

---
Abstract

Neural network quantization is an essential technique for deploying models on resource-constrained devices. However, its impact on model perception, particularly regarding class activation maps (CAMs), remains a significant area of investigation. In this study, we explore how quantization alters the perceptual field of vision models, shedding light on the alignment between CAMs and visual saliency maps across various architectures. Leveraging a dataset of 10,000 images from ImageNet, we rigorously evaluate six diverse architectures: VGG16, ResNet50, EfficientNet, MobileNet, SqueezeNet, and DenseNet. Through systematic quantization techniques applied to these models, we uncover nuanced changes in CAMs and their alignment with visual saliency maps. Our findings reveal the varying sensitivities of different architectures to quantization and underscore its implications for real-world applications in terms of model performance and interpretability. This study contributes to advancing our understanding of neural network quantization, providing insights crucial for deploying efficient and interpretable models in practical settings.


---


[!Table Quantitative Results](imgs/quant_tab.png)


[!Figure Quantitative Results](imgs/quant_graph.png)


[!Figure Qualitative Results](imgs/ILSVRC2012_val_00000002.png)

