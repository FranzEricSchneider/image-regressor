# Built to enable running on an EC2 instance that is almost already there:
# Deep Learning AMI GPU Pytorch 2.0.1 (Amazon Linux 2) 20230627
pip3 install opencv-python
pip3 install torchsummaryX
pip3 install wandb
echo "Run with python3 image_regressor/main.py <directory>"
