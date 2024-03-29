## Scale Guided Hypernetwork for Blind Super-Resolution Image Quality Assessment

[Paper](https://arxiv.org/abs/2306.02398)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=JunFu1995/SGH)

<img src="img/framework.png" width="800px"/>

## TODO
- [x] ~~Code release~~
- [x] ~~Upload datasets~~ 
- [ ] clean the code


## Introduction

* `log`: save training log
* `nets`: define iqa model 
* `save`: save model 
* `nets`: define iqa model 
* `datasets.py`: define datasets 
* `datasets_deepsrq.py`: define datasets for deepsrq
* `engine.py`: training and test engine
* `train_test_IQA.py`: setup training and test 

## Train and Test
First, download datasets used in this paper from [here](https://drive.google.com/drive/folders/13OhtsoWcLZL64BXoAH4NmCCCFgBXywDH?usp=sharing). 

Second, change the dataset path in the `train_test_IQA.py` as follows:
```python
    path = {
        'QADS': 'yourpath/QADS/',
        'CVIU': 'yourpath/CVIU/',
        'Waterloo': 'yourpath/Waterloo/',
    } 
```
Third, train and test the model using the following command:
```
python train_test_IQA.py --dataset xxx --netFile xxx --gpuid x --batch_size 64
```
Some mandatory options:
* `--dataset`: string, Training and testing dataset, support datasets: 'CVIU' | 'QADS'| 'Waterloo'.
* `--netFile`: string, IQA model, support models: 'DBCNN' | 'HyperIQA' | 'CNNIQA' | 'Resnet50' | 'JCSAN' | 'DeepSRQ'.
* `--gpuid`: int, gpu device 
* `--batch_size`: int, Batch size, 64.

## Acknowledgement
This project is based on [HyperIQA](https://github.com/SSL92/hyperIQA). Thanks for the awesome work.

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{fu2023scale,
  title={Scale Guided Hypernetwork for Blind Super-Resolution Image Quality Assessment},
  author={Fu, Jun},
  journal={arXiv preprint arXiv:2306.02398},
  year={2023}
}
```
## Contact
For any questions, feel free to contact: `fujun@mail.ustc.edu.cn`