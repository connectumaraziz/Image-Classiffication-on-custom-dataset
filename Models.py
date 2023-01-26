# Use models code in step:1 of notebook


# For Xception Model
# import
from tensorflow.keras.applications.xception import Xception, preprocess_input
# Use below in Model selection selection Step:1
pre_trained_model = Xception(
    include_top=False,
    weights='imagenet', # Weightfile 
    input_shape=(299, 299, 3))

<<--------------------------------------------------------------------------------------------->>
# For DenseNet Model
# DenseNet Family
# .DenseNet121
# .DenseNet169
# .DenseNet201 
# import
from tensorflow.keras.applications.DenseNet121 import DenseNet121, preprocess_input
# Use below in Model selection selection Step:1
pre_trained_model = DenseNet121(
    include_top=False,
    weights='imagenet', # Weightfile 
    input_shape=(299, 299, 3))


<<-------------------------------------------------------------------------------------------->>
# For ResNet Model
# ResNet Family
# .ResNet50
# .ResNet50V2
# .ResNet101
# .ResNet101V2
# .ResNet152
# .ResNet152V2
# import
from tensorflow.keras.applications.ResNet50 import ResNet50, preprocess_input
# Use below in Model selection selection Step:1
pre_trained_model = ResNet50(
    include_top=False,
    weights='imagenet', # Weightfile 
    input_shape=(299, 299, 3))


<<-------------------------------------------------------------------------------------------->>
# For VGGNet Model
# VGGNet Family
# .VGG16
# .VGG19
# import
from tensorflow.keras.applications.VGG16 import VGG16, preprocess_input
# Use below in Model selection selection Step:1
pre_trained_model = VGG16(
    include_top=False,
    weights='imagenet', # Weightfile 
    input_shape=(299, 299, 3))

<<-------------------------------------------------------------------------------------------->>
# For MobileNet Model
# MobileNet Family
# .MobileNet
# .MobileNetV2
# .MobileNetV3
# import
from tensorflow.keras.applications.MobileNet import MobileNet, preprocess_input
# Use below in Model selection selection Step:1
pre_trained_model = MobileNet(
    include_top=False,
    weights='imagenet', # Weightfile 
    input_shape=(299, 299, 3))

<<-------------------------------------------------------------------------------------------->>
# For Inception Model
# Inception Family
# .InceptionV3
# .InceptionResNetV2
# .MobileNetV3
# import
from tensorflow.keras.applications.InceptionV3 import InceptionV3, preprocess_input
# Use below in Model selection selection Step:1
pre_trained_model = InceptionV3(
    include_top=False,
    weights='imagenet', # Weightfile 
    input_shape=(299, 299, 3))



<<-------------------------------------------------------------------------------------------->>