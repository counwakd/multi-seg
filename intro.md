![DINOI](https://github.com/user-attachments/assets/3aa4123c-f118-4740-9ed1-3cc83857b798)# 2D视觉基础模型梳理

## Vision Transformer (ViT)
- **核心思想**：纯Transformer架构，将输入图像切分为固定大小的patch，嵌入序列并加上位置编码后送入Transformer编码器。
- **ViT 的关键工作原理**
  1. **将图像转换为 Patch 序列**：
     - 将输入图像按固定大小（如 16×16）划分为多个小块（Patch）。例如，对于 224×224 的图像，按 16×16 划分后会得到 196 个 Patch。
     - 每个 Patch 被展平为一维向量，然后通过线性变换映射到指定的维度空间（如 768 维），形成一个 token。

  2. **添加位置编码**：
     - 由于 Transformer 缺少位置信息，因此需要为每个 Patch 添加位置编码（Positional Encoding）。ViT 使用可学习的位置编码向量，并将其直接加到 Patch token 上。

  3. **添加 Class Token**：
     - 在 Patch token 序列前添加一个特殊的 Class Token，用于表示整个图像的全局特征。这个 Class Token 会在 Transformer 编码过程中与其他 token 一起参与计算。

  4. **输入 Transformer Encoder**：
     - 将包含 Class Token 的 Patch token 序列输入到标准的 Transformer 编码器中。Transformer 编码器由多层 Encoder Block 组成，每层包括 Layer Norm、Multi-Head Attention、Dropout 和 MLP Block。

  5. **分类**：
     - Transformer 编码器的输出是一个序列，但 ViT 只使用 Class Token 的输出。将 Class Token 的输出送入 MLP Head，最终得到分类结果。

- **ViT 的模型架构**：
  
  ![image](https://github.com/user-attachments/assets/37fe3dcc-d950-43f2-aa7f-6f9a16f70b92)

  1. **Patch Embedding 层**：
     - 将图像划分为 Patch，并将每个 Patch 映射为一维向量。
      - 使用卷积层实现 Patch 的线性变换，例如使用 16×16 的卷积核，步距为 16，卷积核个数为 768。

  2. **Transformer Encoder**：
     - 由多个 Encoder Block 堆叠而成，每个 Block 包括 Layer Norm、Multi-Head Attention、Dropout 和 MLP Block。

  3. **MLP Head**：
     - 用于分类的全连接层，通常由一个或多个线性层组成。
       
简单VIT代码示意
```
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```
- **应用场景**：图像分类等任务。
- **特点**：
  - 采用全局自注意力机制，能够捕捉长距离依赖关系。
  - 适合大规模数据集训练，性能优异。

## EfficientViT
- **核心思想**：通过高效注意力机制和分层设计，在保证准确率的同时大幅降低推理延迟和计算开销，适合移动端和实时应用。
- **EfficientViT核心改进**：
  
EfficientViT 的主要改进点包括：

  1. **分层结构设计**：
     - 采用类似 CNN 的分层结构（Stage 设计），逐步降低空间分辨率，提升通道数。前几层处理高分辨率特征，后几层提取更强的语义信息。

  2. **高效注意力机制**：
     - 结合局部注意力（Local Attention）和全局 token 汇聚（Global Aggregation）：
       - **局部注意力**：在高分辨率阶段使用局部注意力，降低计算成本。
       - **全局 token 汇聚**：保留全局上下文信息，避免丢失长程依赖。

  3. **轻量化设计**：
     - 大量使用深度可分离卷积（Depthwise Convolution）和分组卷积，进一步降低计算量。
     - 在一些变体中引入线性注意力或 token 稀疏化，减少计算开销。

  4. **多任务适配**：
     - 设计上支持多种任务，包括图像分类、语义分割、目标检测等，尤其在密集预测任务（如分割）上表现优异。

- **EfficientViT模型架构**：
  
EfficientViT 的整体架构可以分为以下几个部分：

![evi](https://github.com/user-attachments/assets/e01e9c1b-20c1-48d9-8243-42430b3ed266)

项目源码 : https://github.com/microsoft/Cream/tree/main/EfficientViT

  1. **Patch Embedding 层**：
     - 将输入图像划分为固定大小的 Patch，并将每个 Patch 映射为一维向量。
     - 使用线性投影将 Patch 转换为指定维度的特征向量。

  2. **分层 Transformer 编码器**：
     - 由多个分层的 Transformer 编码器组成，每层包括：
       - **局部注意力模块**：处理局部特征，降低计算成本。
       - **全局 token 汇聚模块**：保留全局上下文信息。
       - **Feed-Forward Network (FFN)**：进一步处理特征。
       - **Token Interaction**：实现局部和全局特征的交互。

  3. **轻量化模块**：
     - 使用深度可分离卷积和分组卷积，减少计算量。
     - 在一些变体中引入线性注意力或 token 稀疏化。

  4. **多任务适配模块**：
     - 根据不同任务需求，设计相应的头部结构（如分类头、分割头、检测头）。
       
- **主要改进**：
  - 局部注意力+全局token汇聚
  - 大量使用深度可分离卷积与分组卷积
  - 支持多任务适配
- **应用场景**：图像分类、语义分割、目标检测等。
- **特点**：
  - 高效的计算性能，适合资源受限的设备。
  - 灵活的架构设计，支持多种视觉任务。

## DETR
- **核心思想**：将目标检测建模为集合预测问题，使用CNN backbone + Transformer Encoder-Decoder架构，直接预测最终的检测集合。
-  **DETR的工作原理**
  
DETR 的工作原理可以分为以下几个关键步骤：

  1. **CNN Backbone**：
     - 使用卷积神经网络（如 ResNet）提取图像的特征图。这些特征图将作为后续 Transformer 编码器的输入。

  2. **Transformer Encoder**：
     - 将 CNN 提取的特征图展平为一维序列，并添加位置编码（Positional Encoding）以保留空间信息。
     - 输入到 Transformer 编码器中，编码器通过自注意力机制建模特征之间的全局关系。

  3. **Transformer Decoder**：
     - Transformer 解码器使用一组可学习的查询向量（Object Queries），每个查询向量对应一个可能的目标。
     - 解码器通过交叉注意力机制结合编码器的输出和查询向量，逐步细化目标的特征表示。

  4. **预测头（Prediction Head）**：
     - 对每个查询向量的输出，使用全连接层预测目标的类别和边界框。
     - 通过一个分类头预测目标类别，通过一个回归头预测边界框的坐标。

  5. **损失函数**：
     - 使用二分图匹配（Bipartite Matching）将预测结果与真实标注进行匹配，计算分类损失和边界框回归损失。
     - 匹配未成功的目标预测被分配为“无目标”类别。

-  **DETR的模型架构**
  
DETR 的整体架构可以分为以下几个部分：
![dert](https://github.com/user-attachments/assets/606e534e-d3a3-40dd-afd0-db56bce40128)

项目源码： https://github.com/facebookresearch/detr

  1. **CNN Backbone**：
     - 通常使用预训练的卷积神经网络（如 ResNet-50）提取图像的特征图。

  2. **Transformer Encoder**：
     - 输入：将 CNN 特征图展平为一维序列，并添加位置编码。
     - 结构：由多个 Encoder Layer 组成，每个 Layer 包括 Layer Norm、Multi-Head Self-Attention 和 Feed-Forward Network (FFN)。

  3. **Transformer Decoder**：
     - 输入：一组可学习的查询向量（Object Queries）。
     - 结构：由多个 Decoder Layer 组成，每个 Layer 包括 Layer Norm、Multi-Head Self-Attention、Multi-Head Cross-Attention 和 FFN。

  4. **预测头（Prediction Head）**：
     - 分类头：预测目标类别。
     - 回归头：预测边界框的坐标。
- **应用场景**：目标检测。
- **特点**：
  - 消除了传统目标检测中的anchor和NMS等手工设计步骤。
  - 利用Transformer的全局建模能力，提升检测性能。

## DINO (IDEA)
- **核心思想**：在DETR基础上改进，通过对比式去噪训练、混合查询选择和Look Forward Twice等机制提升训练速度和效果。
- **DINO的核心改进**：
 
DINO 的主要改进点包括以下三个方面：

  1. **对比式去噪训练（Contrastive DeNoising Training, CDN）**：
     - 在训练过程中引入带噪声的锚框，包括轻微噪声（正样本）和大噪声（负样本）。
     - 模型需要学会区分并恢复这些噪声框，从而更好地利用标注框信号，避免预测重复框，同时加快收敛速度。

  2. **混合查询选择（Mixed Query Selection）**：
     - 在原始 DETR 中，所有查询均为随机可学习参数，学习效率较低。
     - DINO 中一部分查询从图像特征中动态生成锚框，带有空间先验；另一部分保持可学习，用于捕捉语义信息，从而结合空间位置和语义学习，提升定位精度。

  3. **Look Forward Twice**：
     - 解码器的后几层会产生更准确的框，DINO 利用这些结果来指导前几层的预测。
     - 类似于“自蒸馏”过程，使训练更稳定，减少抖动。

- **DINO的模型架构**：
DINO 的整体架构基于 DETR，包含以下部分：

![DINOI](https://github.com/user-attachments/assets/d676f94c-dd39-4315-b873-b22ca8ad3623)

项目源码： https://github.com/IDEA-Research/DINO

  1. **CNN Backbone**：
   - 使用卷积神经网络（如 ResNet-50 或 Swin Transformer）提取图像特征。
  
  2. **Transformer Encoder-Decoder**：
     - 编码器将图像特征序列化并建模全局关系。
     - 解码器包含一组可学习的查询（queries），每个查询都去图像中寻找可能的目标。

  3. **预测头（Prediction Head）**：
     - 对每个查询的结果，输出目标的类别和边界框。

  4. **改进模块**：
     - CDN 模块：在训练阶段引入带噪样本。
     - 查询选择模块：整合在解码器的查询初始化中。
     - Look Forward Twice：将后层梯度信息反馈给前层优化。
- **应用场景**：目标检测。
- **特点**：
  - 更快的收敛速度和更高的检测精度。
  - 适合大规模数据集训练。

## CLIP
- **核心思想**：双塔模型，图像编码器和文本编码器协同训练，通过对比学习让匹配的图文对嵌入相似度最大化。
- **CLIP 的工作原理**
  
CLIP 的工作原理可以分为以下几个关键步骤：

  1. **双塔模型架构**：
     - **图像编码器**：使用卷积神经网络（如 ResNet）或 Vision Transformer（ViT）将图像编码为固定维度的向量。
     - **文本编码器**：使用 Transformer 架构将文本（如描述或标签）编码为固定维度的向量。
  
  2. **对比学习**：
     - 将图像和文本编码器的输出嵌入到共享的嵌入空间中。
     - 通过对比学习（Contrastive Learning）最大化匹配的图文对的相似度，最小化不匹配的图文对的相似度。
     - 使用 InfoNCE 损失函数来优化模型，使得匹配的图文对在嵌入空间中更接近，不匹配的图文对更远离。
  
  3. **零样本分类**：
     - 在测试阶段，模型可以利用预训练的文本嵌入来构建分类器，从而实现零样本分类。
     - 例如，给定一组类别标签的文本描述，模型可以将测试图像嵌入到嵌入空间中，并计算其与每个类别标签嵌入的相似度，从而预测图像的类别。

- **CLIP 的模型架构**

CLIP 的整体架构可以分为以下几个部分：

![CLIP](https://github.com/user-attachments/assets/b557607b-7ea7-4d0c-936b-bff395833096)

项目源码： https://github.com/openai/CLIP

  1. **图像编码器**：
     - 使用卷积神经网络（如 ResNet）或 Vision Transformer（ViT）将图像编码为固定维度的向量。
     - 例如，使用 ViT-B/32，将图像编码为 512 维的向量。
  
  2. **文本编码器**：
     - 使用 Transformer 架构将文本编码为固定维度的向量。
     - 例如，使用一个 12 层的 Transformer，将文本编码为 512 维的向量。
  
  3. **对比学习模块**：
     - 将图像和文本嵌入到共享的嵌入空间中。
     - 使用 InfoNCE 损失函数进行对比学习，优化模型参数。
  
  4. **零样本分类模块**：
     - 在测试阶段，利用预训练的文本嵌入构建分类器，实现零样本分类。
- **应用场景**：图文检索、零样本图像分类等。
- **特点**：
  - 利用自然语言监督，实现跨模态对齐。
  - 无需大量标注数据，具有很强的泛化能力。

## BLIP
- **核心思想**：统一理解与生成，引入captioning模块提升训练数据质量，支持图文匹配、检索、图像描述、VQA等任务。
- **BLIP 的核心创新**
 
BLIP 的主要创新点包括：

  1. **混合编码器-解码器架构（MED）**：
     - 提出了一种用于有效多任务预训练和灵活迁移学习的新模型架构，包含三个子模块：
       - **单模态编码器**：分别对图像和文本进行编码。
       - **图像引导的文本编码器**：通过交叉注意力层注入视觉信息。
       - **图像引导的文本解码器**：用于生成基于图像的文本描述。
  
  2. **高效利用噪声数据**：
     - 提出了一种“字幕和过滤”（CapFilt）的方法，先使用噪声数据训练 BLIP，再利用预训练的 Captioner 生成字幕，通过 Filter 过滤噪声，最后使用干净的数据重新训练。

- **BLIP 的模型架构**
 
BLIP 的整体架构可以分为以下几个部分：

![BLIP](https://github.com/user-attachments/assets/f7418b2e-fa69-40c1-839c-86b18f6cd5e8)

项目源码：https://github.com/salesforce/BLIP

  1. **图像编码器**：
     - 使用 Vision Transformer（ViT）作为图像编码器，将输入图像分成 patch，编码为一系列 Image Embedding。
  
  2. **文本编码器**：
     - 使用 BERT 作为文本编码器，对文本进行编码，并在文本输入的开头添加特殊标记以总结句子。
  
  3. **图像引导的文本编码器**：
     - 在文本编码器的每个 Transformer 块中插入交叉注意力层，注入视觉信息，用于图像-文本匹配任务。
  
  4. **图像引导的文本解码器**：
     - 用因果自注意力层替换文本编码器中的双向自注意力层，用于生成基于图像的文本描述。
  
  5. **预训练目标**：
     - 联合优化三个目标：
       - **图像-文本对比损失（ITC）**：对齐视觉和文本表示。
       - **图像-文本匹配损失（ITM）**：预测图像-文本对是否匹配。
       - **语言建模损失（LM）**：生成基于图像的文本描述。
         
- **应用场景**：图文匹配、图像描述、视觉问答等。
- **特点**：
  - 结合了理解与生成任务，提升模型的多功能性。
  - 自举式数据清理，提高训练数据质量。

## BLIP-2
- **核心思想**：使用冻结的预训练视觉模型和轻量中间模块Q-Former，将视觉特征转化为适合LLM的输入，实现高效的视觉-语言结合。
- **BLIP-2 的核心架构**
  
BLIP-2 的核心架构包括三个主要部分：

![BLIP2](https://github.com/user-attachments/assets/c58d88d4-2dba-471e-a29b-c862e66bb073)

项目源码 : https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models

  1. **冻结的图像编码器**：
     - 通常使用类似 CLIP 的视觉模型（如 ViT），其权重在训练过程中保持不变。
  
  2. **轻量级的查询 Transformer（Q-Former）**：
     - 这是 BLIP-2 的核心模块，用于桥接图像编码器和语言模型之间的模态差距。
     - Q-Former 包含两个子模块：
       - **图像 Transformer**：用于从冻结的图像编码器中提取视觉特征。
       - **文本 Transformer**：作为文本编码器和解码器，用于生成文本。
  
  3. **冻结的大型语言模型（LLM）**：
     - 如 Flan-T5 或 LLaMA，其权重在训练过程中也保持不变。

- **BLIP-2 的预训练策略**
  
BLIP-2 采用两阶段预训练策略：

1. **视觉语言表示学习**：
   - 使用冻结的图像编码器，通过图像-文本对比学习（ITC）、图像-文本匹配（ITM）和图像引导的文本生成（ITG）任务，学习视觉与语言的对齐。
   - ITC 任务通过对比图像和文本嵌入来对齐模态。
   - ITM 任务通过二分类任务判断图像和文本是否匹配。
   - ITG 任务通过生成文本任务来学习图像特征。

2. **视觉到语言的生成学习**：
   - 将 Q-Former 提取的视觉特征传递给冻结的 LLM，通过语言建模损失进行预训练。
   - 对于解码器型 LLM，使用语言建模损失进行训练。
   - 对于编码器-解码器型 LLM，使用前缀语言建模损失进行训练。

- **应用场景**：图文对齐、图文生成、指令微调等。
- **特点**：
  - 利用大型语言模型的强大生成能力，提升多模态任务的性能。
  - Q-Former作为视觉和语言之间的桥梁，实现高效的特征转换。

## GLIP
- **GLIP 的核心思想**
  
GLIP 的核心思想包括以下几点：

  1. **统一目标检测和短语定位**：
     - GLIP 将目标检测任务重新构想为短语定位问题，通过联合训练图像编码器和语言编码器来预测区域和单词的正确配对。
     - 这种统一框架允许 GLIP 同时从检测数据和接地数据中学习，提升两项任务的性能。
  
  2. **利用大量图像-文本对进行预训练**：
     - GLIP 使用了 2700 万图像-文本对进行预训练，其中包括 300 万人工标注数据和 2400 万网络爬取的数据。
     - 通过自训练方式生成定位框，使学习到的表示具有丰富的语义信息。
  
  3. **强大的零样本和少样本迁移能力**：
     - GLIP 在预训练后展示了强大的零样本和少样本迁移能力，即使在没有直接见过特定数据集的情况下，也能在新的对象检测任务上表现出色。

- **GLIP 的模型架构**
  
GLIP 的整体架构包括以下几个部分：

![GLIP](https://github.com/user-attachments/assets/64264ed5-0191-4ca3-b6d7-fca800f52eba)

项目源码： https://github.com/microsoft/GLIP

  1. **图像编码器**：
     - 使用 Swin Transformer 作为图像编码器，提取图像的区域特征。
  
  2. **文本编码器**：
     - 使用 BERT 作为文本编码器，提取文本提示的语义特征。
  
  3. **深度融合检测头**：
     - 检测头模块包含文本编码模块和交叉注意力机制，负责将文本特征与图像特征进行深度融合。
     - 通过跨模态多头注意力模块（X-MHA），在多个编码层中反复引入交叉注意力，增强两模态特征的交互。

- **应用场景**：目标检测、短语定位、图文对比学习等。
- **特点**：
  - 结合目标检测和语言理解，提升模型的语义理解能力。
  - 适合开放词汇目标检测任务。

## SAM
- **SAM 的核心思想**
  
SAM 的核心思想包括以下几点：

  1. **可提示分割（Promptable Segmentation）**：
     - 用户可以通过简单的提示（如点击一个点、绘制一个框、输入一段文本）来获取目标的分割掩码。
     - 这种设计统一了不同类型的分割任务，包括语义分割、实例分割和交互分割。
  
  2. **大规模数据驱动**：
     - SAM 在一个大规模的分割数据集（SA-1B，包含 11 亿个分割掩码）上进行预训练，使其具备强大的泛化能力。
  
  3. **零样本分割能力**：
     - SAM 具备零样本分割能力，可以在从未见过的图像和任务上直接应用，无需额外训练。

- **SAM 的模型架构**
  
SAM 的整体架构包括三个主要模块：

![SAM](https://github.com/user-attachments/assets/c64fc939-5aec-48a3-b73a-c61281ee0e32)

项目源码 : https://github.com/facebookresearch/segment-anything

  1. **图像编码器（Image Encoder）**：
     - 使用 Vision Transformer（ViT-Huge）作为图像编码器，将整张图像编码为稠密特征图。
     - 这个特征图可以被多次重用，提高了计算效率。
  
  2. **提示编码器（Prompt Encoder）**：
     - 输入：点、框、多边形、文本等提示。
     - 将这些提示编码为向量（或稀疏 token），用于引导分割。
  
  3. **掩码解码器（Mask Decoder）**：
     - 一个轻量级的 Transformer 解码器，输入图像特征和提示特征，输出分割掩码。
     - 每个掩码还会配一个 IoU 置信度，方便选择最佳掩码。
       
- **应用场景**：交互式分割、自动分割、零样本迁移等。
- **特点**：
  - 通用性强，支持多种类型的分割任务。
  - 大规模数据驱动，具备强大的泛化能力。

## DINO (Meta)
- **核心思想**：自监督训练，学生网络模仿教师网络输出，通过多尺度视图训练和特定正则化方法学习视觉特征。
- **DINO 的核心原理**
DINO 的核心原理包括以下几点：

  1. **教师-学生框架**：
     - 教师模型和学生模型共享相同的网络结构（通常是 Vision Transformer, ViT），但它们的参数并不绑定。教师模型负责提供指导信号，而学生模型则通过优化目标函数逐步逼近教师模型的行为。
  
  2. **中心化策略**：
     - DINO 提出了一个独特的中心化操作，用于稳定训练过程并提高收敛速度。通过对特征向量进行中心化处理，可以减少冗余信息的影响，使模型更加关注重要的模式。
  
  3. **温度控制机制**：
     - 温度超参被引入到 softmax 函数中，以调整概率分布的锐利程度。较低的温度值可以使分布更集中，有助于增强相似样本之间的关联性；反之，则鼓励探索更多样化的表达形式。
  
  4. **互信息最大化原则**：
     - 学生模型的目标是最小化自身输出与经由动平均更新后的教师模型之间 KL 散度损失项。这一过程实际上相当于实现了跨视图间一致性的保持以及局部上下文中细粒度关系的学习。

- **DINO 的模型架构**
  
DINO 的整体架构包括以下几个部分：

项目源码 : https://github.com/facebookresearch/dinov3

1. **Vision Transformer (ViT)**：
   - 使用 ViT 作为骨干网络，能够从图像中提取丰富的语义信息。

2. **自蒸馏机制**：
   - 通过自蒸馏的方式，让模型在训练过程中自我优化，无需预训练的教师模型。

3. **多尺度视图训练**：
   - 教师模型接收两张全局视图，学生模型接收这两张全局视图和多张小局部视图，从而提升模型对不同尺度特征的理解。
- **应用场景**：无监督视觉特征学习。
- **特点**：
  - 无需标注数据，利用自监督学习提取鲁棒的视觉特征。
  - 适合多种视觉任务的基础特征提取。

## DINOv2
- **核心思想**：在DINO基础上扩展数据规模和训练策略，引入patch-level自监督训练和新的训练技巧，得到更鲁棒的视觉特征。
- **DINOv2 的核心改进**
  
DINOv2 的主要改进点包括以下几点：

  1. **大规模数据集**：
     - 使用了 1.42 亿张图像的 LVD-142M 数据集进行训练，这些图像通过自监督图像检索技术精心筛选，提高了数据集的质量和多样性。
  
  2. **训练方法**：
     - 采用了一种区分性自监督方法（Discriminative Self-supervised Pre-training），可以看作是 DINO 和 iBOT 损失以及 SwAV 中心化的组合。
  
  3. **多任务支持**：
     - DINOv2 不仅支持线性分类、深度估计、图像检索等下游任务，还能在无需微调的情况下直接应用于多种视觉任务。

- **DINOv2 的模型架构**
  
DINOv2 的整体架构包括以下几个部分：

项目源码 : https://github.com/facebookresearch/dinov2

  1. **Vision Transformer (ViT)**：
     - 使用 ViT 作为骨干网络，能够从图像中提取丰富的语义信息。
  
  2. **自监督学习**：
     - 通过自监督学习，DINOv2 无需任何标注数据即可学习高质量的视觉特征。
  
  3. **多尺度视图训练**：
     - 在训练过程中，DINOv2 使用多尺度视图来提升模型对不同尺度特征的理解。

- **DINOv2 的性能**
  
DINOv2 在多个基准数据集上表现出色，例如：

- **语义分割**：在 ADE20k 数据集上，DINOv2 的线性探针 mIoU 达到 49.5。
- **深度估计**：在 NYUv2 数据集上，DINOv2 的 RMSE 为 0.37。
- **3D 关键点匹配**：在 NAVI 数据集上，DINOv2 的召回率为 60.1%。

- **DINOv2 与 DINOv3 的对比**
  
DINOv3 是 DINOv2 的后续版本，它在多个方面进行了改进和扩展。以下是 DINOv2 和 DINOv3 的主要对比：

| 特性 | DINOv2 | DINOv3 |
|------|--------|--------|
| **数据集大小** | 1.42 亿张图像 | 17 亿张图像 |
| **参数量** | 11 亿参数 | 70 亿参数 |
| **语义分割性能** | ADE20k mIoU 49.5 | ADE20k mIoU 55.9 |
| **深度估计性能** | NYUv2 RMSE 0.372| NYUv2 RMSE 0.309 |
| **3D 关键点匹配** | NAVI 召回率 60.1% | NAVI 召回率 64.4% |

- **DINOv2 的训练方法**
  
DINOv2 的训练方法包括以下几个关键步骤：

  1. **图像嵌入计算**：
     - 使用预训练的 ViT-H/16 自监督神经网络计算每个图像的嵌入。
  
  2. **k-means 聚类**：
     - 采用 k-means 聚类算法将嵌入向量相似的图像放到同一聚类中。
  
  3. **数据集筛选**：
     - 从 12 亿张图片中筛选出 1.42 亿张高质量图像，形成 LVD-142M 数据集。
  
  4. **自监督预训练**：
     - 使用区分性自监督方法进行预训练，结合 DINO 和 iBOT 损失以及 SwAV 中心化。
       
- **应用场景**：多种视觉任务的通用特征提取。
- **特点**：
  - 更大规模的数据集和更精细的训练策略。
  - 支持高分辨率输入，提升对小尺度目标的检测能力。

## DINOv3
- **核心思想**：大规模自监督学习，解决稠密特征退化问题，支持高分辨率和密集特征的稳定训练。
- **DINOv3 的核心改进**
  
DINOv3 的主要改进点包括以下几点：

  1. **大规模数据集**：
     - 使用了 17 亿张图像的 LVD-142M 数据集进行训练，这些图像通过自监督图像检索技术精心筛选，提高了数据集的质量和多样性。
  
  2. **模型规模扩展**：
     - 将模型参数从 DINOv2 的 11 亿扩展到 70 亿，增强了模型的表示能力。
  
  3. **Gram Anchoring**：
     - 引入了 Gram Anchoring 技术，通过约束学生模型与早期教师模型的 Gram 矩阵一致，保留局部特征一致性，解决长训练中密集特征退化的问题。
  
  4. **简化训练策略**：
     - 采用恒定学习率与权重衰减，避免复杂超参数调度，大幅提升训练稳定性。
  
  5. **高效蒸馏框架**：
     - 创新多学生并行蒸馏管道，允许同时训练多个学生模型并在所有训练节点共享教师推理，显著节省计算资源。

- **DINOv3 的模型架构**
  
DINOv3 的整体架构包括以下几个部分：

项目源码 : https://github.com/facebookresearch/dinov3

  1. **Vision Transformer (ViT)**：
     - 使用 ViT 作为骨干网络，能够从图像中提取丰富的语义信息。
  
  2. **自监督学习**：
     - 通过自监督学习，DINOv3 无需任何标注数据即可学习高质量的视觉特征。
  
  3. **多尺度视图训练**：
     - 在训练过程中，DINOv3 使用多尺度视图来提升模型对不同尺度特征的理解。
     
- **应用场景**：图像分类、分割、目标检测等。
- **特点**：
  - 通过GramAnchoring等技术解决稠密特征退化问题。
  - 支持多种视觉任务，具备强大的泛化能力。

## Grounding DINO
- **核心思想**：将物体检测任务统一为语言驱动的定位任务，通过深度融合视觉和语言信息实现零样本检测能力。
- **Grounding DINO 的核心架构**
  
Grounding DINO 的架构设计如下：

![Grounding DINO](https://github.com/user-attachments/assets/3113a044-a44f-46d0-acdd-db9d32936bcb)

项目源码 : https://github.com/IDEA-Research/GroundingDINO

  1. **双编码器-单解码器架构**：
     - **图像骨干网络**：使用如 Swin Transformer 的网络提取多尺度图像特征。
     - **文本骨干网络**：使用如 BERT 的文本编码器提取文本特征。
     - **特征增强器**：融合图像和文本特征，使用可变形自注意力处理图像特征，普通自注意力处理文本特征，并包含图像到文本和文本到图像的交叉注意力模块。
     - **语言引导的查询选择模块**：根据输入文本选择与之更相关的特征作为解码器的输入。
     - **跨模态解码器**：结合图像和文本特征，预测目标的边界框坐标和提取对应的标签。
  
  2. **子句级文本特征**：
     - 引入子句级文本特征以避免不必要的词间交互，使得模型能够同时编码多个类别名称，同时避免不同类别之间的不必要依赖。
       
- **应用场景**：开放词汇目标检测。
- **特点**：
  - 语言引导的查询选择，提升检测效率和准确性。
  - 跨模态融合，实现强大的零样本检测能力。

## Grounding DINO 1.5 & 1.6
- **核心思想**：在Grounding DINO基础上进行版本迭代和性能优化，提升语义理解能力和检测效率。
- **Grounding DINO 1.5 & 1.6 的核心改进**
  
  ### Grounding DINO 1.5

  项目源码 : https://github.com/IDEA-Research/Grounding-DINO-1.5-API
  
    - **Pro 版**：
      - 使用更大的模型和更多的数据进行训练，提升了检测精度。
      - 在大规模数据集构建和高精度需求场景中表现出色。
    - **Edge 版**：
      - 针对端侧部署进行了优化，提升了推理速度。
      - 在 NVIDIA Orin NX 卡上实现了 10FPS 的推理速度。
      - 适用于具身智能、自动驾驶等需要实时目标检测的场景。
  
  ### Grounding DINO 1.6
    - **Pro 版**：
      - 使用更大规模、更高质量的训练数据集（Grounding-30M），在零样本迁移基准测试中取得了新的 SOTA 结果。
      - 在 COCO 数据集上达到 55.4 AP，在 LVIS-minival 数据集上达到 57.7 AP。
    - **Edge 版**：
      - 进一步优化了模型架构，提升了推理速度。
      - 在输入尺寸为 800×800 时，推理速度达到 14FPS，相比 1.5 Edge 版提升了 40%。
      - 在 COCO 数据集上达到 44.8 AP，在 LVIS-minival 数据集上达到 34.6 AP。

- **应用场景**：开放词汇目标检测。
- **特点**：
  - 使用更大的架构和更丰富的数据集，提升模型性能。
  - 优化网络结构，提高推理速度和效率。

## GroundedSAM
- **核心思想**：结合开放词汇检测（Grounding DINO）与零样本分割（SAM），实现文本提示驱动的检测与分割。
- **GroundedSAM 的核心架构**
  
GroundedSAM 的整体架构包括以下几个部分：

![Grounded SAM](https://github.com/user-attachments/assets/1b42b626-357f-4ec3-97bc-e602f7d77b0b)

项目源码 : https://github.com/IDEA-Research/Grounded-SAM-2

  1. **Grounding DINO**：
     - 作为开放集目标检测器，根据文本输入生成目标的检测框。
     - 使用如 Swin Transformer 的骨干网络提取图像特征，并结合文本特征进行目标定位。
  
  2. **SAM（Segment Anything Model）**：
     - 作为分割模型，根据检测框生成对应的像素级掩码。
     - 使用 Vision Transformer（ViT）作为图像编码器，将整张图像编码为稠密特征图。
  
  3. **提示编码器（Prompt Encoder）**：
     - 将文本提示编码为向量，用于引导分割模型生成特定目标的掩码。
  
  4. **掩码解码器（Mask Decoder）**：
     - 根据图像特征和提示特征，输出目标的分割掩码。
     - 
- **应用场景**：开放世界的检测与分割，自动标注、图像编辑等复杂视觉任务。
- **特点**：
  - 强大的跨模态能力，支持多种复杂视觉任务。
  - 可扩展性强，可与其他模型组合实现更多功能。

## DINO-X
- **核心思想**：开放世界统一视觉模型，支持多模态Prompt输入和多任务输出，实现检测、分割、姿态估计、语言理解等一体化。
- **DINO-X 的核心架构**
  
DINO-X 的架构设计如下：

项目源码 : https://github.com/IDEA-Research/DINO-X-API

  1. **Transformer 编码器-解码器架构**：
     - 采用与 Grounding DINO 1.5 相同的基于 Transformer 的编码器-解码器架构，以追求用于开放世界目标理解的对象级表示。
  
  2. **灵活的提示选项**：
     - 支持三种类型的提示：文本提示、视觉提示和自定义提示。
     - **文本提示**：根据用户提供的文本输入识别所需对象。
     - **视觉提示**：用户可以通过框选或点选图中的一个物体，AI 能找出图中所有同类物体。
     - **自定义提示**：用户可以“教”模型识别特定领域的物体，如医疗图像中的器官。
  
  3. **通用对象提示**：
     - 开发了一个通用的对象提示来支持无需提示的开放世界检测，使得无需用户提供任何提示即可检测图像中的任何物体成为可能。

- **DINO-X 的技术特点**
  - **强大的开放世界目标检测能力**：DINO-X 能够识别已知和未知物体，并进行更深层次的理解和分析。
  - **亿级数据训练**：构建了包含超过 1 亿个高质量定位样本的大规模数据集 Grounding-100M，用于提升模型的开放词汇表检测性能。
  - **多功能集成**：DINO-X 集成了多个感知头，以同时支持多个对象感知和理解任务，包括检测、分割、姿态估计、对象字幕、基于对象的问答等。
  - **两个版本，各有所长**：
    - **DINO-X Pro**：旗舰版，精度最高，适合对性能要求高的场景。
    - **DINO-X Edge**：轻量版，优化速度，在边缘设备上达到更快的推理速度，适合实时应用。
      
- **应用场景**：多种视觉任务的综合应用。
- **特点**：
  - 灵活的输入和丰富的输出，支持多种视觉任务。
  - 集成多种任务头，实现一体化的视觉理解。

## LLaVA
- **核心技术**
  - **模型架构**

    ![LLaVA](https://github.com/user-attachments/assets/4c055c68-5e9f-4791-a937-45f61d2ab8fc)

  项目源码 : https://github.com/haotian-liu/LLaVA

    - LLaVA 采用了独特的多模态融合架构，将文本信息和视觉信息相结合，通过深度学习算法对海量数据进行训练，使模型能够更好地理解和生成与场景相关的内容。
    - 其架构基于 Transformer 等先进架构进行优化，具有强大的并行计算能力和扩展性，能够处理复杂的任务。
  
  - **训练方法**
    - 采用强化学习和监督学习相结合的训练方式，通过大量的标注数据和反馈机制，让模型不断优化自身的参数，提高准确性和鲁棒性。
    - 在训练过程中，还引入了数据增强技术，如对图像进行旋转、缩放等操作，对文本进行同义词替换等，以增强模型的泛化能力。

- **应用场景**
  - **自然语言处理**
    - 在文本生成方面，LLaVA 可以根据用户输入的提示生成高质量的文章、故事、诗歌等，语言流畅自然，逻辑清晰。
    - 在语言翻译领域，能够实现多种语言之间的准确翻译，支持实时翻译功能，为跨语言交流提供便利。
  
  - **图像识别与生成**
    - 对于图像识别，LLaVA 能够快速准确地识别图像中的物体、场景和人物，并进行分类和标注。
    - 在图像生成方面，可以根据用户提供的描述生成相应的图像内容，如根据文字描述生成虚拟场景、角色形象等，为创意设计和艺术创作提供灵感。

- **优势**
  - **高效性**
    - LLaVA 的计算效率高，在处理大规模数据和复杂任务时能够快速给出结果，节省时间和资源。
  
  - **准确性**
    - 经过精心训练和优化，其在各种任务中的准确率都达到了行业领先水平，能够为用户提供可靠的结果。
  
  - **易用性**
    - 提供了简洁易用的接口和工具，无论是开发者还是普通用户，都可以方便地使用 LLaVA 的功能，无需复杂的操作和专业知识。

