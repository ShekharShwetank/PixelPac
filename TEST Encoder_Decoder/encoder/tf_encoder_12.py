import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input, Model
def lightweight_block(input_tensor):
    # Depthwise convolution (preserves input dimensions)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(input_tensor)
    
    # 1x1 convolution to mix channels and increase to 8
    x = layers.Conv2D(8, kernel_size=1, padding='same')(x)
    
    # 3x3 convolution block with BatchNorm and ReLU
    x = layers.Conv2D(8, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "A")(x)
    
    return x

def downsample_block(input_tensor):
    # Depthwise convolution with stride=2 for downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 1x1 convolution for channel mixing
    x = layers.Conv2D(input_tensor.shape[-1], kernel_size=1, padding='same')(x)
    
    # Second depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "B")(x)
    
    return x

def double_channels_same_dim_block(input_tensor):
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 1x1 convolution to double the channels
    out_channels = input_tensor.shape[-1] * 2
    x = layers.Conv2D(out_channels, kernel_size=1, padding='same')(x)
    
    # Second depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "C")(x)
    
    return x

def downsample_same_channels_block(input_tensor):
    # Depthwise conv with stride=2 for downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 1x1 conv for channel mixing (same channels)
    channels = input_tensor.shape[-1]
    x = layers.Conv2D(channels, kernel_size=1, padding='same')(x)
    
    # Second depthwise conv
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "D")(x)
    
    return x

def double_channels_block(input_tensor):
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out_channels = input_tensor.shape[-1] * 2
    x = layers.Conv2D(out_channels, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "E")(x)

    return x

def downsample_block_same_channels(input_tensor):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    channels = input_tensor.shape[-1]
    x = layers.Conv2D(channels, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "F")(x)

    return x

def double_channels_block_22(input_tensor):
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out_channels = input_tensor.shape[-1] * 2
    x = layers.Conv2D(out_channels, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "G")(x)

    return x

def downsample_block_2(input_tensor):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    channels = input_tensor.shape[-1]
    x = layers.Conv2D(channels, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "H")(x)

    return x

def double_channels_block_2(input_tensor):
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out_channels = input_tensor.shape[-1] * 2
    x = layers.Conv2D(out_channels, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "I")(x)

    return x


#################################### upsampling and concat blocks ####################################

def fuse_and_double_channels_block(low_res_input, high_res_input):
    # Upsample the low-res input (11, 19, 128) to match high-res spatial dims (22, 38)
    low_res_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(low_res_input)
    
    # Concatenate along channel axis -> shape becomes (22, 38, 64 + 128 = 192)
    x = layers.Concatenate(axis=-1)([high_res_input, low_res_upsampled])
    
    # Standard Conv2D to reduce channels from 192 to 96
    x = layers.Conv2D(96, kernel_size=3, padding='same')(x)
    
    # Depthwise separable processing
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(96, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name = "J")(x)

    return x

def fuse_without_conv_block(low_res_input, high_res_input):
    # Upsample the low-res input (22, 38, 96) to match high-res spatial dims (44, 76)
    low_res_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(low_res_input)
    
    # Concatenate along channel axis -> shape becomes (44, 76, 96 + 32 = 128)
    x = layers.Concatenate(axis=-1)([high_res_input, low_res_upsampled])
    
    # Depthwise conv + BN + ReLU
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 1x1 conv for channel mixing to reduce channels to 40
    x = layers.Conv2D(40, kernel_size=1, padding='same')(x)
    
    # Another depthwise conv + BN + ReLU
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="K")(x)
    
    return x

def fuse_and_adjust_channels_block(low_res_input, high_res_input):
    # Upsample the low-res input (44, 76, 32) to match high-res spatial dims (88, 152)
    low_res_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(low_res_input)
    
    # Concatenate along channel axis -> shape becomes (88, 152, 32 + 16 = 48)
    x = layers.Concatenate(axis=-1)([high_res_input, low_res_upsampled])
    
    # Standard Conv2D to increase channels from 48 to 58
    x = layers.Conv2D(58, kernel_size=3, padding='same')(x)
    
    # Depthwise separable processing
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(58, kernel_size=1, padding='same')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="L")(x)

    return x

def fuse_without_conv_block_v2(low_res_input, high_res_input):
    # Upsample the low-res input (88, 152, 58) to match high-res spatial dims (176, 304)
    low_res_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(low_res_input)
    
    # Concatenate along channel axis -> shape becomes (176, 304, 58 + 8 = 66)
    x = layers.Concatenate(axis=-1)([high_res_input, low_res_upsampled])
    
    # Depthwise conv + BN + ReLU
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 1x1 conv for channel mixing to get output channels = 64
    x = layers.Conv2D(64, kernel_size=1, padding='same')(x)

    # Another depthwise conv + BN + ReLU
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="out")(x)

    return x


inputlayer = layers.Input(shape= (176,304,3))
skip_connections = []
outputs = lightweight_block(inputlayer)
skip_connections.append(outputs)

outputs = downsample_block(outputs)
outputs = double_channels_same_dim_block(outputs)
skip_connections.append(outputs)

outputs = downsample_same_channels_block(outputs)
outputs = double_channels_block(outputs)
skip_connections.append(outputs)

outputs = downsample_block_same_channels(outputs)
outputs = double_channels_block_22(outputs)
skip_connections.append(outputs)

outputs = downsample_block_2(outputs)
outputs = double_channels_block_2(outputs)

A, C, E, G = skip_connections
#concate I(==outputs) and H 
J = fuse_and_double_channels_block(outputs, G)
#concate J and E
K = fuse_without_conv_block(J, E)
#concate K and C
L = fuse_and_adjust_channels_block(K, C)
# concate L and A
out = fuse_without_conv_block_v2(L, A)

model = Model(inputs = inputlayer, outputs = out)

import time
for i in range(10):
    input_tensor = tf.random.normal([1,176,304,3])
    start  = time.time()
    output_data = model(input_tensor)
    print(f"time for operations is : {(time.time() - start)*1000.:2f} ms")
