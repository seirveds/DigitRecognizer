# Neural digit recognition

This project uses a flask webserver to provide interactive input for digit recognition using neural networks. The two neural networks implemented are a deep multilayer perceptron and a convolutional neural network based on [LeNet-5](<http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>). 



# Requirements

- Python 3.6
- numpy 1.16.2
- Flask 1.0.2
- Keras 2.2.4
- python-mnist 0.6
- Pillow 5.3.0

# How to run

Running main.py starts a webserver on <http://127.0.0.1:5000/>. Inside main.py the variable ```MODEL_TYPE```can be changed to switch between the convolutional neural network (default) and the deep mlp. 

# How to use

Once the server has been started, go to <http://127.0.0.1:5000/>, draw a single digit in the input box and hit the classify button. The neural network will try to classify the digit drawn.