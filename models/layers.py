import tensorflow as tf
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings("ignore")

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()

        #multi head attention block form keras
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)
        #feedforward layer
        self.ffn = tf.keras.Sequential([
            layers.Dense(feed_forward_dim, activation='relu'),
            layers.Dense(embedding_dim)
        ])
        #normalization and dropout
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        #forward pass for TrnsformerEncodeblock
        attn_output = self.att(x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
    

class PatchTransformerEncoder(tf.keras.Model):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4, num_layers=4):
        super(PatchTransformerEncoder, self).__init__()

        #creating the transformerEncoder from TransformerEncoderBlock
        self.transformer_layers = [
            TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, feed_forward_dim=1024)
            for _ in range(num_layers)
        ]

        self.embedding_convPxP = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            use_bias=False
        )
        
        #generating the traininable positional encodings
        self.positional_encodings = self.add_weight(
            shape=(500, embedding_dim),
            initializer='random_normal',
            trainable=True
        )
        

    def call(self, x):
        # x: [batch, C, H, W] expected in TF
        print(f"Input shape : {x.shape}")
        x = self.embedding_convPxP(x)  # [batch, embedding_dim, H', W']
        print(f"embedding shape : {x.shape}")
        batch_size, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        x = tf.reshape(x, [batch_size, h * w, c])  # [batch, tokens, embedding_dim]
        positional =  self.positional_encodings[:h * w]  # broadcast positional encodings
        print(f"positionla embedding shape before: {positional.shape}")

        x += positional
        x = tf.transpose(x, perm=[1,0,2])
        print(f"positionla embedding shape after : {positional.shape}")

        for layer in self.transformer_layers:
            x = layer(x)
        
        print(f"transformer output shape: {x.shape}")
        return x  # [batch, tokens, embedding_dim]
    
class PixelWiseDotProduct(tf.keras.Model):
    def call(self, x, K):
        # x: [batch, H, W, C], K: [batch, out_channels, C]
        batch_size, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_flat = tf.reshape(x, [batch_size, h * w, c])  # [batch, HW, C]
        x_flat = tf.transpose(x_flat, [0, 2, 1])        # [batch, C, HW]
        K_T = tf.transpose(K, [0, 2, 1])                # [batch, C, out_channels]

        y = tf.matmul(tf.transpose(x_flat, [0, 2, 1]), K_T)  # [batch, HW, out_channels]
        y = tf.transpose(y, [0, 2, 1])
        return tf.reshape(y, [batch_size, -1, h, w])  # [batch, out_channels, H, W]
    

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf

    # Set seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Dummy input: [batch, height, width, channels] => [1, 40, 40, 3]
    dummy_input = tf.random.normal([1, 40, 40, 3])

    # Initialize encoder
    encoder = PatchTransformerEncoder(in_channels=3, patch_size=10, embedding_dim=128, num_heads=4, num_layers=4)

    # Forward pass
    encoded_output = encoder(dummy_input)  # Shape: [batch, tokens, embedding_dim]
    print("Encoded output shape:", encoded_output.shape)

    # Reshape encoded output into [batch, height, width, channels]
    # With patch size 10 and input size 40x40 → 4x4 = 16 tokens
    encoded_reshaped = tf.reshape(encoded_output, [1, 4, 4, 128])
    print(f"encoded_reshaped : {encoded_reshaped.shape}")

    # Create dummy K: [batch, out_channels, embedding_dim]
    K = tf.random.normal([1, 64, 128])

    # Initialize and apply pixel-wise dot product
    dot_product = PixelWiseDotProduct()
    output = dot_product(encoded_reshaped, K)
    print("Pixel-wise dot product output shape:", output.shape)


