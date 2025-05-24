import tensorflow as tf

# Step 1: Define or import your mViT model
from miniVit import mViT


if __name__ == "__main__":
    # Step 2: Create and save the model
    model = mViT(in_channels=3, patch_size=10)
    dummy_input = tf.random.normal([1, 40, 40, 3])
    _ = model(dummy_input)  # Run once to build
    print(_)
    # # Save the model
    # tf.saved_model.save(model, "mvit_saved_model")
    # print("✅ Saved the TensorFlow model.")

    # # Step 3: Convert to TFLite
    # converter = tf.lite.TFLiteConverter.from_saved_model("mvit_saved_model")
    # tflite_model = converter.convert()

    # # Save the TFLite model
    # with open("mvit_model.tflite", "wb") as f:
    #     f.write(tflite_model)
    # print("✅ Converted and saved the TFLite model.")

    # # Step 4: Load and test the TFLite model
    # print("\n=== TensorFlow Lite mViT Model Test ===")

    # # Load the model
    # interpreter = tf.lite.Interpreter(model_path="mvit_model.tflite")
    # interpreter.allocate_tensors()

    # # Get input/output details
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # # Prepare dummy input
    # dummy_input = tf.random.normal([1, 40, 40, 3]).numpy()
    # interpreter.set_tensor(input_details[0]['index'], dummy_input)

    # # Run inference
    # interpreter.invoke()

    # # Get outputs
    # output = interpreter.get_tensor(output_details[0]['index'])
    # attn = interpreter.get_tensor(output_details[1]['index']) if len(output_details) > 1 else None

    # print("Output shape (probabilities):", output.shape)
    # print("Attention map shape:", attn.shape if attn is not None else "Not Available")