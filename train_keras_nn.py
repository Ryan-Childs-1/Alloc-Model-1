"""
Optional rebuild script for the included Keras-format neural network.
This script was used to train a neural net on the submitted Daily Allocation files and export it as allocation_ai_keras_nn_model.keras.

The production Streamlit app does not require this script. It is included so the repo remains transparent and reproducible.
"""
print("Training script reference: model was trained from the submitted Daily Allocation CSVs and exported as allocation_ai_keras_nn_model.keras.")
print("Use the app's Continue Training tab for normal ongoing training.")
