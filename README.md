# PyTorch Implementation of Joint Face Detection and Alignment Multi-task Cascaded Convolutional Networks 

### Step 1:
```Shell
mkdir prepare_data/Data/
mkdir prepare_data/Combined_Data/
mkdir model_store
```

### Step 2:
```Shell
export PYTHONPATH=$PYTHONPATH:(Path of Directory)
```

## Training PNet:

### Step 3:
```Shell
python prepare_data/wider_face_bbox_generate.py
python prepare_data/combined_data_pnet.py
python training/train_p_net.py
```

## Training RNet

### Step 4:
```Shell
python prepare_data/rnet_generate.py
python prepare_data/combined_data_rnet.py
python training/train_r_net.py
```

## Training ONet

### Step 5:
```Shell
python prepare_data/onet_generate.py
python prepare_data/combined_data_onet.py
python training/train_o_net.py
```

## Inference

```Shell
python visualize.py
```

### References
Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.
