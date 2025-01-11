# FedProto: Federated Prototype Learning across Heterogeneous Clients

本程序是对论文 [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243) 的复现工作，主要使用了原论文的开源代码(https://github.com/yuetan031/fedproto)，并对其进行了一定的修改。

## Requirments
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5
* tensorboard

## Data Preparation
请自行下载数据集MNIST和CIFAR10并放在data文件夹下。

目录结构：

- data/mnist/MNIST
- data/cifar10/cifar-10-batches-py

本工作已经准备好在cifar10上训练时所需要的data/resnet/resnet18-5c106cde.pth

## Running the experiments
例如：

python ./exps/federated_main.py --mode task_heter --dataset mnist --num_classes 10 --num_users 20 --ways 3 --shots 100 --stdev 2 --rounds 100 --train_shots_max 110 --ld 1

## Options
各种传递给实验的参数的默认值可以在```options.py```中找到。以下是一些参数的详细信息：

* ```--dataset:```  默认值：'mnist'。可选值：'mnist'，'femnist'，'cifar10'
* ```--num_classes:```  默认值：10。可选值：10，62，10
* ```--mode:```     默认值：'task_heter'。可选值：'task_heter'，'model_heter
* ```--seed:```     随机种子。默认值设为1234。
* ```--lr:```       默认学习率设为0.01
* ```--momentum:```       默认动量设为0.5。
* ```--local_bs:```  默认本地批次大小设为4。


#### Federated Parameters
* ```--mode:```     默认值：'task_heter'。可选值：'task_heter'，'model_heter
* ```--num_users:``` 用户数量。默认值为20
* ```--ways:```      平均本地类数。默认值为3。
* ```--shots:```      每个本地类的平均样本数。默认值为100
* ```--test_shots:```     每个本地类的平均测试样本数。默认值为15
* ```--ld:```      原型损失的权重。默认值为1。
* ```--stdev:```     标准差。默认值为1。
* ```--train_ep:``` 每个用户的本地训练轮次。默认值为1。


#### others
* ```--SSL:```     是否启用半监督学习及相应损失函数。若不开启，则半监督相关的参数无效。默认值为False
* ```--pu_weight:```     半监督学习损失函数的权重。默认值为0.5。
* ```--positiveRate:```    带标签的样本比率。默认值为0.5。
* ```--record_file:``` 训练产生的数据及最终结果的存储位置。默认值为 records.json。
* ```--use_UH:```  模型是否使用固定头。默认值为 False。


## Citation

复现工作主要参考文献：

```
@inproceedings{tan2021fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}

@article{lin2022federated,
    title={Federated Learning with Positive and Unlabeled Data},
    author={Xinyang Lin and Hanting Chen and Yixing Xu and Chao Xu and Xiaolin Gui and Yiping Deng and Yunhe Wang},
    journal={INTERNATIONAL CONFERENCE ON MACHINE LEARNING, VOL 162},
    year={2022}
}

@article{dai2023tackling,
    title={Tackling Data Heterogeneity in Federated Learning with Class Prototypes},
    author={Yutong Dai and Zeyuan Chen and Junnan Li and Shelby Heinecke and Lichao Sun and Ran Xu},
    journal={arXiv preprint arXiv:2212.02758},
    year={2023}
}
```