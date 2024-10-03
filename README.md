# Qwen2 VL 72B AWQ Gradio UI

This repo provides a simple Gradio UI to run Qwen2 VL 72B AWQ.

## Setup

1. Create a virtual environment:
   ```
   python3 -m venv qwen_venv
   ```

2. Activate the virtual environment:
   ```
   source qwen_venv/bin/activate
   ```

3. Install requirements:
   ```
   pip install transformers accelerate qwen-vl-utils gradio
   pip install flash-attn --no-build-isolation
   ```

## Running the Model

1. Activate the virtual environment:
   ```
   source /home/kurwa/Desktop/qwen_venv/bin/activate
   ```

2. Run the script:
   ```
   python run_qwen_model.py --flash-attn2
   ```

This will start the Gradio interface, allowing you to interact with the Qwen2-VL-72B-AWQ model through a web browser.

Note: You need to download the Qwen2-VL-72B-AWQ model separately from Hugging Face.
