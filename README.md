# hello-tensor-handwritten

## set environment

Installing with virtualenv
Take the following steps to install TensorFlow with Virtualenv:

Start a terminal (a shell). You'll perform all subsequent steps in this shell.
1. Install pip and virtualenv by issuing the following commands:

 $ sudo easy_install pip
 $ sudo pip install --upgrade virtualenv 

2. Create a virtualenv environment by issuing a command of one of the following formats:

 $ virtualenv --system-site-packages ~/hello-tensor-handwritten # for Python 2.7

3. Activate the virtualenv environment by issuing one of the following commands:

 $ source ~/hello-tensor-handwritten/bin/activate

4. Ensure pip ≥8.1 is installed:

 (tensorflow)$ easy_install -U pip

5. install TensorFlow and all the packages that TensorFlow requires into the active Virtualenv environment:

 (tensorflow)$ pip install --upgrade tensorflow 



# Execute example

(tensorflow)$ python digits.py


# sources:

* [www.tensorflow.org/install/install_mac](https://www.tensorflow.org/install/install_mac)



