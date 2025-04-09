# IGAF

## Data
The method is evaluated on:   

MVTec AD dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)   

VisA dataset: [VisA](https://github.com/amazon-science/spot-diff/)    

BTAD dataset: [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)   


### MVTec AD

```
📁 data
 └── 📁 mvtec_anomaly_detection
     ├── 📁 bottle
     │   ├── 📁 ground_truth
     │   ├── 📁 test
     │   └── 📁 train
     ├── 📁 cable
     └── ...
```

### VisA

```
📁 data
 └── 📁 visa
     ├── 📁 candle
     │   ├── 📁 ground_truth
     │   ├── 📁 test
     │   └── 📁 train
     ├── 📁 capsules
     └── ...
```


### BTAD

```
📁 data
 └── 📁 btad
     ├── 📁 01
     │   ├── 📁 ground_truth
     │   ├── 📁 test
     │   └── 📁 train
     ├── 📁 02
     └── ...
```


## Quick start


```
conda create -n <your_env> python=3.8
conda activate <your_env>
pip install -r requirements.txt
```


### Training
```
python train.py --gpu_id 0 --obj_id -1 --lr 0.03 --bs 16 --epochs 800 --data_path /your/dataset/path/mvtec_ad/ --log_path /your/log/path/checkpoints_mvtecad/ --checkpoint_path /your/checkpoints/path/checkpoints_mvtecad/ --visualize 
```


### Testing
```
python test.py --gpu_id 0 --base_model_name base_model_name --data_path /your/dataset/path/mvtec_ad/ --checkpoint_path /your/checkpoints/path/checkpoints_mvtecad/
```



















