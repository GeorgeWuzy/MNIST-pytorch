# MNIST-pytorch

A simple handwriting network for handwritten digit string recognition with Pytorch. 

对于邮编图片的数字识别，选用深度学习分类的方法对手写邮编数字进行识别。首先使用Opencv对邮政编码中的数字图片进行提取，形成多个手写数字的图片。卷积网络选用CNN卷积神经网络和ResNet卷积神经网络，基于Pytorch框架在MNIST数据集上进行手写数字识别的训练，损失函数loss选用交叉熵损失，优化器选用Adam，训练完成使用模型对手写数字进行识别并进行可视化实现。最终识别的效果良好。

1.train.py对网络进行训练，可选取CNN和ResNet

2.model.py包含了手写的CNN和ResNet网络

3.demo.py为MNIST数据集测试可视化

4.demo_stamp.py为针对邮票字符串的手写数字识别demo
