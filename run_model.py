# run_model.py

import torch
import torch._dynamo as dynamo
from microcode import AIProgram  
from transformers import AutoModelForCausalLM
from lowering import lower_fx_graph

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "Qwen/Qwen3-4B"  # or "meta-llama/Llama-3-7B"
SEQ_LEN = 16                  # dummy input sequence length
DUMMY_BATCH = 1
OUTPUT_FILE = "instruction_stream.txt"

# ---------------------------
# WRAPPER TO RETURN ONLY TENSORS
# ---------------------------
class TracedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Disable past_key_values caching and only return logits tensor
        out = self.model(x, use_cache=False)
        return out.logits


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    # Wrap the model for safe TorchDynamo export
    model = TracedWrapper(model)

    print("Creating dummy input...")
    dummy_input = torch.randint(0, 100, (DUMMY_BATCH, SEQ_LEN))

    print("Tracing model with TorchDynamo export...")
    gm = dynamo.export(model, dummy_input)[0]

    print("Lowering FX Graph to AI_Accelerator ISA...")
    program = AIProgram()
    program = lower_fx_graph(gm, program)

    print(f"Saving instruction stream to {OUTPUT_FILE} ...")
    program.save_txt(OUTPUT_FILE)

    print("Done!")


# ---------------------------
# ENTRYPOINT
# ---------------------------
if __name__ == "__main__":
    main()

