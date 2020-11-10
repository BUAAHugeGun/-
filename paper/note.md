## SinGAN: Learning a Generative Model from a Single Natural Image

sinGan

用噪声一级一级地生成图像，每级都进行判别

## Specifying Object Attributes and Relations in Interactive Scene Generation

作者提出了一种从给定的场景图生成图像的方法。将图像构造布局与物体外观分割开来。通过双重嵌入使生成的图像可以更好地匹配输入的场景图，同时这个嵌入方法还可以从同一个场景图中生成多个不同的输出图像，用户也能控制这个生成过程。目前项目已开源：http://github.com/ashual/scene_generation。

## Image Generation from Layout  

对每个框生成m+n维向量，其中n维向量对G来说是标准正态分布，对GT来说是CNN预测的正态分布。

多个物体进行CNN+cLSTM融合成一个输出，用G生成。

## Cross-domain Correspondence Learning for Exemplar-based Image Translation

风格图+mask图，从固定噪声生成

## Panoptic-based Image Synthesis  

nvidia

全景感知卷积、上采样

$x=W^T(x\cdot M)\frac{sum(1)}{sum(M)}+b,M=1-clip(abs(x-x_{center}))_{[0,1]}$

语义图、全景图

## Semantically Multi-modal Image Synthesis

更换风格，encoder+decoder，encoder输出潜变量z，测试时采样

GroupDNet、Conditional Group Block(CG-Block) 

https://github.com/Seanseattle/SMIS

## Semantic Image Synthesis with Spatially-Adaptive Normalization

GauGAN， SPADE

空间自适应归一化层（Spatially-Adaptive Normalization Layer）

https://github.com/NVlabs/SPADE

## Local Class-Specific and Global Image-Level Generative Adversarial Networks for Semantic-Guided Scene Generation  

本文解决的是语义场景生成任务。在全局图像级别生成方法中，一个挑战是生成小物体和细致局部的纹理。为此这项工作考虑在局部上下文中学习场景生成，并相应地设计一个以语义图为指导、局部的特定类生成网络，该网络分别构建和学习专注于生成不同场景的子生成器，并能提供更多场景细节。为了学习更多的针对局部生成的、有辨识力的类特定表征，还提出了一种新颖的分类模块。为了结合全局图像级别和局部特定类生成的优势，设计了一个联合生成网络，其中集成了注意力融合模块和双判别器结构。https://github.com/Ha0Tang/LGGAN

## Controlling Style and Semantics in Weakly-Supervised Image Generation

spade++

## Photographic Image Synthesis with Cascaded Refinement Networks

crn，没用gan

## Image-to-Image Translation with Conditional Adversarial NetWorks

conditional gan(CGAN)

## Semi-parametric Image Synthesis

1. 根据大致的草图框架（语义布局法），深度神经网络现在可以直接合成真实效果的图片。
2. 参数模型（parametric models）的优点是具有高度的表现力，可进行端到端的训练。非参数模型（Non-parametric models）的优点是可以在测试时提取大型的真实图片数据集里的素材。
3. 集结这两种方法的优势，提出了SIMS（半参数模型）
4. SIMS工作思路：
   （1）先用大型真实图像数据集先训练非参数模型，相当于获得了一个合成素材库；
   （2）然后基于语义布局（Semantic layout），把这些素材填充进去，就像一张图被分割成好几个版块之后，再往上打补丁充实细节。接缝的地方，深度网络会自行融合，并计算好版块之间物体的空间关系，进一步加强视觉的真实效果。

## High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

**方法**：1、coarse-to-fine generator（G1,G2）

   2、multi-scale 判别器

   3、除了对抗损失和感知损失加入了feature matching loss

   4、除了segmantic label map 加入了instance map（相同类的不同对象相邻时无法区分）

 

**结果：**1、分辨率为2048*1024

   2、可在原始标签map中改变标签（如将建筑替换成树木）

   3、允许用户编辑单个对象的外观（如：汽车外观和路面纹理）

   4、更为逼真的纹理和细节

   5、同样的label map输入可以得到多样性的结果

 

**概念：**1、使用语义分割方法，可以将图像转换到语义标签域，在标签域中编辑对象，然后转换回图像域。

## Tell, Draw, and Repeat: Generating and Modifying Images Based on Continual Linguistic Instruction  
