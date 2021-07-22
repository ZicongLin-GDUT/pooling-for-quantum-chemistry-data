# pooling-for-quantum-chemistry-data
The source code of paper Different Pooling Methods in Graph Neural Network for Large Scale Quantum Chemistry Data. 

## Requirements
The **Python environment** used for the experiment in this paper is as follows:
- Python  3.6.13
- CUDA  11.2
- Pytorch  1.8.0
- cudatoolkit  11.1
- torch-geometric  1.7.0
    - torch-cluster  1.5.9 
    - torch-scatter  2.0.6
    - torch-sparse  0.6.9
    - torch-spline-conv  1.2.1
- rdkit  2021.03.1
- tensorboard  2.5.0
- ogb  1.3.1

The **configuration of the computer** used in the experiment is as follows:
- System: Linux (Ubuntu 18.04 LTS)
- CPU: Intel(R) Xeon(R) W-2223 @3.60GHz
- Memory: 32GB
- GPU: GeForce RTX 3080 (10 GB)

## Usages
If you want to replicate the experiment in this paper, you can use the following command-line code (use GIN-based model with mean pooling as an example):
```python
python main_demo.py --gnn gin --pooling_method global --graph_pooling mean --residual True --JK concat --num_workers 4 --log_dir './log' --checkpoint_dir './ckpt' --save_test_dir './result'
```
Explanation and default values for all command-line parameters can be found in `main_demo.py`.

The following code can be used to check your training process:
```python
from tensorboard.backend.event_processing import event_accumulator

# open the log file
ea = event_accumulator.EventAccumulator(file_path)  # Change file_path to your file path
ea.Reload()
# read data
train = ea.scalars.Items('train/mae')
valid = ea.scalars.Items('valid/mae')
train_mae = [epoch.value for epoch in train]
valid_mae = [epoch.value for epoch in valid]
```
