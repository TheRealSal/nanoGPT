import torch

def load_stolen_weights(model, stolen_ckpt_path):
    ckpt = torch.load(stolen_ckpt_path, map_location="cpu")
    W = ckpt["weights"]  # [50257, n_embd] float32 typically

    with torch.no_grad():
        # nanoGPT vocab_size may be 50304; stolen is often 50257
        V_stolen = W.shape[0]
        V_model, D_model = model.transformer.wte.weight.shape
        assert W.shape[1] == D_model, (W.shape, (V_model, D_model))

        # copy into first V_stolen rows
        model.transformer.wte.weight[:V_stolen].copy_(W.to(model.transformer.wte.weight.device))

        # optional sanity check: ensure tying still holds
        assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()

    print(f"Loaded stolen embeddings from {stolen_ckpt_path} into wte (tied lm_head).")