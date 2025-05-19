import tensorflow as tf
from tensorflow.keras import layers, Model
from layers import PatchTransformerEncoder, PixelWiseDotProduct

class mViT(tf.keras.Model):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()

        self.norm = norm
        self.n_query_channels = n_query_channels

        self.patch_transformer = PatchTransformerEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=3,
            strides=1,
            padding='same'
        )

        self.regressor = tf.keras.Sequential([
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dense(dim_out)
        ])

    def call(self, x):
        # x: [batch, height, width, channels]
        tgt = self.patch_transformer(tf.identity(x))  # [batch, tokens, embedding_dim]

        x = self.conv3x3(x)  # [batch, height, width, embedding_dim]

        regression_head = tgt[:, 0, :]  # [batch, embedding_dim]
        queries = tgt[:, 1:self.n_query_channels + 1, :]  # [batch, n_query_channels, embedding_dim]

        # Pixel-wise dot product: x is [batch, h, w, embedding_dim], queries is [batch, n_query_channels, embedding_dim]
        range_attention_maps = self.dot_product_layer(x, queries)  # [batch, n_query_channels, h, w]

        y = self.regressor(regression_head)  # [batch, dim_out]

        if self.norm == 'linear':
            y = tf.nn.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return tf.nn.softmax(y, axis=1), range_attention_maps
        else:
            y = tf.nn.sigmoid(y)

        y = y / tf.reduce_sum(y, axis=1, keepdims=True)
        return y, range_attention_maps


if __name__ == "__main__":
    print("=== TensorFlow mViT Model Test ===")
    
    model = mViT(in_channels=3, patch_size=10)

    dummy_input = tf.random.normal([2, 40, 40, 3])  # [batch, height, width, channels]
    
    output, attn = model(dummy_input)

    print("Output shape (probabilities):", output.shape)      
    print("Attention map shape:", attn.shape)                 
