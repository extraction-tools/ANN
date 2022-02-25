#Define custom loss function to use here...
def custom_loss(y_true, y_pred):
  loss = K.square(y_true - y_pred)
  loss = K.sum(loss)
  return loss


#Example usage in Keras model compilation
"""globalModel.compile(
    #Parameters: Optimizer, Learning_rate, (loss function), epochs,
    #Adjust the learning rate here
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(.0001),
    #loss = tf.keras.losses.MeanSquaredError(),
    loss = custom_loss
)"""
