import tensorflow as tf
from tensorflow.keras import layers, models
from miniVit import mViT
from encoder_decoder import LiteEncoderDecoder
# from encoder_decoder import Custom_Encoder_Decoder

class UnetAdaptiveBins(tf.keras.Model):
    def __init__(self, backend, n_bins = 256, min_val = 0.1, max_val = 10, norm = "linear"):
        super(UnetAdaptiveBins, self).__init__()

        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val

        self.encoder_decoder = LiteEncoderDecoder()
        self.adaptive_bins_layers= mViT(
                                    in_channels=3,
                                    n_query_channels=64,
                                    patch_size=16,
                                    dim_out=n_bins,
                                    embedding_dim=64,
                                    num_heads=4,
                                    norm='linear'
                                        )
        self.conv_out= models.Sequential([
                                    layers.Conv2D(filters=n_bins, kernel_size=1, strides=1, padding='valid'),
                                    layers.Softmax(axis=3)  # Softmax applied on the channel dimension 
                                ])
        
    def call(self, x):
        #get the output from encoder decoder block
        # x -> 480*480*3
        # print(f"UnetAdaptiveBins input to encode : {x.shape}")

        enocder_decoder_dummy = tf.random.normal([1,240,240,64])
        extracted_feature_map = enocder_decoder_dummy
    
        encoder_decoder_out = self.encoder_decoder.model(x)

        # print(f"UnetAdaptiveBins output from decoder : {enocder_decoder_dummy.shape}")

        # extracted_feature_map ->240*240*64
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layers(extracted_feature_map)
        # print(f"UnetAdaptiveBins bin_widths_normed, range_attention_maps : {bin_widths_normed.shape}, {range_attention_maps.shape}")
        range_attention_maps = tf.transpose(range_attention_maps, [0, 2, 3, 1])  # (1, 240, 240, 64)
        out = self.conv_out(range_attention_maps)
        # print(f"UnetAdaptiveBins out : {out.shape} ")

        # print(f'extracted_feature_map : {extracted_feature_map.shape}  \nbin_widths_normed, range_attention_map , {bin_widths_normed.shape}, {range_attention_maps.shape}')
        # NOTE ****** [batch, n_query_channels, height, width] IS THE SHAPE OF THE RANGE ATTENTION MAP
        # getting the probabilites for each pixel

        #when performing below the channel dimension changes to n_bins: 
        # print(f"output shape is : {out.shape}")

        # processing the bins
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        # print(f"UnetAdaptiveBins bin_widths : {bin_widths.shape}")
        # bin_widths_normed: shape [batch_size, n_bins]
		# After scaling, we prepend one column filled with self.min_val, just like padding (1, 0) in PyTorch.

        min_vals = tf.fill([tf.shape(bin_widths)[0], 1], self.min_val)
        bin_widths = tf.concat([min_vals, bin_widths], axis=1)
        bin_edges = tf.math.cumsum(bin_widths, axis=1)
        # print(f"UnetAdaptiveBins bin_edges after cumsum: {bin_edges.shape}")

        # Compute bin centers as the average of consecutive bin edges
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])  # shape: [batch, n_bins]
        # print(f"UnetAdaptiveBins centers : {centers.shape}")

        # Reshape centers to [batch, n_bins, 1, 1] for broadcasting
        centers = tf.reshape(centers, [tf.shape(centers)[0], tf.shape(centers)[1], 1, 1])
        # print(f"UnetAdaptiveBins centers (after reshape) : {centers.shape}")

        # Permute 'out' from [batch, H, W, channels] to [batch, channels, H, W] to match shape for multiplication
        out_permuted = tf.transpose(out, [0, 3, 1, 2])  # [B, n_bins, H, W]
        # print(f"UnetAdaptiveBins out_permuted : {out_permuted.shape}")

        # Weighted sum over the bin/channel dimension
        pred = tf.reduce_sum(out_permuted * centers, axis=1, keepdims=True)  # [B, 1, H, W]
        return bin_edges, pred
def test_inference(model, num_iter = 10):
    import time
    for i in range(num_iter):
        dummy = tf.random.normal([1, 240, 240, 3])
        start  =time.time()
        bins, pred = model(dummy)
        print(f"time taken is : {(time.time() - start)*1000.:2f}")
        
if __name__ ==  "__main__":
    dummy = tf.random.normal([1, 240, 240, 3])

    model = UnetAdaptiveBins(100)
    
    bins, pred = model(dummy)
    print(f"shape of bins is : {bins.shape}")
    print(f"shape of pred is : {pred.shape}")

    test_inference(model, num_iter=10)




