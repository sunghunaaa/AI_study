mport tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus) #[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


if(gpus) :
  print("gpus on")
else:
  print("gpus off")
