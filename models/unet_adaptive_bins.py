import tensorflow as tf
from tensorflow.keras import layers, models
from miniVit import mViT

class UpSampleBN(tf.keras.Model):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self.conv_block = tf.keras.Sequential([
            layers.Conv2D(output_features, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(output_features, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])

    def call(self, x, concat_with):
        x_up = tf.image.resize(x, size=tf.shape(concat_with)[1:3], method='bilinear')
        # print(f"x_up shape : {x_up.shape}")
        x_concat = tf.concat([x_up, concat_with], axis=-1)  # TF uses channels_last
        # print(f"x_concat : {x_concat.shape}")
        # return self.conv_block(x_concat)
        output = self.conv_block(x_concat)
        return output


class DecoderBN(tf.keras.Model):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        self.conv2 = layers.Conv2D(num_features, 1, padding='same')

        self.up1 = UpSampleBN(num_features + 112 + 64, num_features // 2)
        self.up2 = UpSampleBN(num_features // 2 + 40 + 24, num_features // 4)
        self.up3 = UpSampleBN(num_features // 4 + 24 + 16, num_features // 8)
        self.up4 = UpSampleBN(num_features // 8 + 16 + 8, num_features // 16)

        self.conv3 = layers.Conv2D(num_classes, 3, padding='same')

    def call(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out


class Encoder(tf.keras.Model):
    def __init__(self, base_model):
        super(Encoder, self).__init__()
        self.base_model = base_model

    def call(self, x):
        features = [x]
        for layer in self.base_model.layers:
            x = layer(x)
            features.append(x)
        return features


class UnetAdaptiveBins(tf.keras.Model):
    def __init__(self, encoder, n_bins=100, min_val=0.1, max_val=10.0, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = encoder

        # 💡 Build mViT internally here like in PyTorch
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins, embedding_dim=128, norm=norm)
        
        self.decoder = DecoderBN(num_classes=128)

        self.conv_out = tf.keras.Sequential([
            layers.Conv2D(n_bins, 1, padding='valid'),
            layers.Softmax(axis=-1)
        ])

    def call(self, x):
        features = self.encoder(x)
        unet_out = self.decoder(features)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = tf.pad(bin_widths, [[0, 0], [1, 0]], constant_values=self.min_val)
        bin_edges = tf.math.cumsum(bin_widths, axis=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        centers = tf.reshape(centers, [-1, self.n_bins, 1, 1])

        pred = tf.reduce_sum(out * centers, axis=1, keepdims=True)
        return bin_edges, pred

    @classmethod
    def build(cls, n_bins, input_shape=(480, 640, 3), norm='linear'):
        base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet', input_shape=input_shape)
        encoder = Encoder(base_model)
        model = cls(encoder, n_bins=n_bins, norm=norm)
        print("Model built successfully.")
        return model
    

if __name__ == '__main__':
    # #------------------ test code for upsample block 
    # x_tf = tf.random.normal([1, 16, 16, 64])        # [batch, height, width, channels]
    # skip_tf = tf.random.normal([1, 32, 32, 64])
    # model_tf = UpSampleBN(skip_input=128, output_features=64)  # 64 + 64 channels

    # out_tf = model_tf(x_tf, skip_tf)
    # print("TensorFlow Output shape:", out_tf.shape)
    

    # model = UnetAdaptiveBins.build(100)
    # x = tf.random.normal([2, 480, 640, 3])
    # print(model.adaptive_bins_layer)
    # bins, pred = model(x)
    # print("Bin edges shape:", bins.shape)
    # print("Prediction shape:", pred.shape)