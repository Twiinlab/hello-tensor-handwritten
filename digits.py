import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r) 
    plt.show()   

# print("Display some digits")
# raw_input('Press enter to continue: ')
# print("after display")
# display(0)
# display(1)
# display(8)
# print len(data[0])

print("Fit a Linear Classifier")
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=1000)

print("Evaluate accuracy")
classifier.evaluate(test_data, test_labels)
print classifier.evaluate(test_data, test_labels)["accuracy"]

print("Classify an example")
print classifier.predict(test_data[1])
# print("Predicted %d, Label: %d" % (classifier.predict(test_data[0]), test_labels[0]))

# print("Visualize learned weights")
# weights = classifier.weights_
# f, axes = plt.subplots(2, 5, figsize=(10,4))
# axes = axes.reshape(-1)
# for i in range(len(axes)):
#     a = axes[i]
#     a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
#     a.set_title(i)
#     a.set_xticks(()) # ticks be gone
#     a.set_yticks(())
# plt.show()
