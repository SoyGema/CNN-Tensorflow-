
# CNN structure for reading a 64x64 minimap 

def cnn_read_fn(features, labels, mode):
  'Model for CNN'
  #Input Layer
  input_layer = tf.reshape(features['x'], [-1, 64, 64, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
    inputs=input_layers,
    filters=32,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu)

  #Pooling Layer 1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Poolig Layer #2
  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],  
    padding='same',
    activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3 and pooling Layer #3
  conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool3_flat = tf.reshape(pool3, [-1, 16 * 16 * 128])
  dense = tf.layers.dense(inputs=pools2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  
