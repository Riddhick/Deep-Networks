
# Deep Networks(PyTorch)

This repository contains implementation of Deep Neural Networks using Pytorch.

### Models Available: 
- LeNet(5) : A CNN with 2 Convolutional Layers and 3 Fully Connected layer. Modified for accepting colour images. 

- AlexNet(Transfer Learned): A transfer learning model on pretrained AlexNet. Can classify two classes- Bees & Ants.


## Network Diagrams


    LeNet(
    (cn1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (pool1): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (cn2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (pool2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (fc1): Linear(in_features=400, out_features=120, bias=True)
    (fc2): Linear(in_features=120, out_features=84, bias=True)
    (fc3): Linear(in_features=84, out_features=10, bias=True)
    )
    
    AlexNet(
        ...
        (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=9216, out_features=4096, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
        )
    )
#### Input Dimensions :

| Network | Dimensions |
|:---:|:---:|
| Lenet   | (3x32x32)  |
| AlexNet | (3x224x224)|