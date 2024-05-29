<h1 align = "center">Lab1: Pretrained CNN Fine-tuning for Bird Recognition</h1>

---

---



<h2 align = "center">1. 项目总览</h2>



​	**本项目使用 Pytorch 深度学习框架，进行了鸟类细粒度识别任务的迁移学习实验。选用了在 ImageNet 预训练的 ResNet 系列模型，在 CUB-200-2011 鸟类图片数据集上进行模型微调。**



- 项目主体包含两个 jupyter notebook 文件`FineTuning.ipynb` , `New_model.ipynb`，分别对应对 ResNet 系列预训练模型进行迁移学习和使用随机初始化的 ResnNet 模型在CUB-200-2011 鸟类图片数据集上从零开始训练。

- `weights/`目录下保存了作者在微调模型和训练新模型的过程中得到的表现较好的模型权重文件。可供在对应 notebook 文件中直接加载权重并测试。
- `dataset/` 目录下保存了作者训练过程中随机分割出的测试集对应的索引文件等。
- `model_logs/search/` 目录下保存了作者在微调模型过程中进行超参数网格搜索得到的不同参数组合对应的 Tensorboard 日志。
- `model_logs/improve/` 目录下保存了作者在得到较优的超参数组合后进一步尝试减少过拟合提升模型表现过程中的训练日志。
- `model_logs/new_models/`  目录下保存了作者在 CUB-200-2011 鸟类图片数据集上从零训练鸟类细粒度分类模型过程中的训练日志。





---

---



<h2 align = "center">2. 使用说明</h2>



#### (1) 数据集准备

- 本 repo 并不包含 CUB-200-2011 鸟类图片数据集文件，请自行下载数据集并按照 .ipynb 文件中 dataset 对象的 root 参数将数据集目录放置在你的项目目录中的指定位置。（***请注意，如果不按照指定目录位置放置数据集文件，并使用 dataset 目录下的 .pth 文件加载数据集，就不能保证你所加载的已有模型权重训练过程中所接触到的数据与你在本地进行测试时所使用的测试集是隔离的。***）

```python
# 加载数据
dataset = datasets.ImageFolder(root='../CUB_200_2011/images')

```

- 如果不希望使用已有的数据集分割，你可以在 .ipynb 文件的 “**数据预处理与数据集分割**” 部分选择测试集的比例，重新随机分割数据集。（***注意，在本地重新分割数据集后如果再次允许 .ipynb 文件请注释掉 `save_datasets(dataset, train_dataset, test_dataset)` 以保证测试集不再改变。***）



####  (2) 预训练模型加载与修改( For `FineTuning.ipynb`)

- 在 `MyResNet` 类中，支持对 *Resnet* 系列模型的选择，以及冻结层数的选择。你可以选择注释掉冻结参数的部分代码以选择冻结参数的比例。

```python
        # # 冻结部分层的参数
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = False
        # for param in self.resnet.layer3.parameters():
        #     param.requires_grad = False
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = False
        # for param in self.resnet.layer5.parameters():
        #     param.requires_grad = False
```



#### (3) 模型训练

- **超参数**：模型训练部分提供了对多种超参数的便捷设定，下面是具体说明：

  - `chosen_model`： 在 Resnet 系列模型中进行选择。 *（支持 resnet 18, 34, 50 ,101 ,152）*

  - `batch_size`: 定义训练采用的 batch_size。***（作者训练过程中受到GPU内存限制设定为 32 ）***

  - `num_epoches`: 定义训练的总轮数

  - `log_freq`: 定义训练过程中每个多少个 iter 记录一次训练集和验证集上的 Loss 和 Accuracy。

  - `train_val_split`: 定义训练集和验证集的比例。***（测试集在最开始已经分割并保存索引）***
  - `regularization_list`： L2 正则化的强度系数。

  - **Learning rate**：
    - 在 `Finetuning.ipynb` 中，支持 `lr_new_list` 和 `lr_old_list`，分别对应新初始化的线性层的学习率和微调已有的未冻结的模型权重使用的学习率。（*设为 lr_list 是为了方便超参数网格搜索的过程。*）
    - 在 `New_model.ipynb` 中，支持  `lr_list` ，即整个模型训练的学习率。

  - `saved_name`：自定义保存训练日志文件的目录名称。

- **优化器**：为了简化训练的流程，作者选用自适应学习率 Adam 优化器，减少了设计学习率衰减策略等工作。
- ***如果选择在本地训练，请在训练完成后保存权重文件到指定目录***：

```python
torch.save(model.state_dict(), 'weights/' + saved_name + '.pth')
```



#### (4) 模型测试

- 加载作者提供的模型权重或本地训练得到的权重，允许测试部分代码即可得到模型在测试集上的分类准确率。


- 加载权重后运行测试部分代码，获得所加载的模型在测试集上的准确率。





