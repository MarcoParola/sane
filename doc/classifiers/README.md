# Classifiers

We condider two distinct architectures: ResNet and ViT.

### ResNet

| layer type  | # layers | trainable params | in the sequence |
|-------------|----------|------------------|-----------------|
| Convolution |  20      |  11159232        | ✅             |
| Batch norm  |  60      |  19200           |   ❌           |
| Linear      |  1       |  5130            |  ✅            |

### ViT

| layer type  | # layers | trainable params | in the sequence |
|-------------|----------|------------------|-----------------|
| Convolution |  1       |  590592          | ✅              |
| Layer norm  |  25      |  38400           |   ✅            |
| Linear      |  49      |  85025290        |  ✅             |