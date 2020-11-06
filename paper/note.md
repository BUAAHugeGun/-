## Specifying Object Attributes and Relations in Interactive Scene Generation  

sinGan

用噪声一级一级地生成图像，每级都进行判别

## Image Generation from Layout  

对每个框生成m+n维向量，其中n维向量对G来说是标准正态分布，对GT来说是CNN预测的正态分布。

多个物体进行CNN+cLSTM融合成一个输出，用G生成。

## Cross-domain Correspondence Learning for Exemplar-based Image Translation

风格图+mask图，从固定噪声生成

