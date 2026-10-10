# Copy the raw per_layer_token_embd tensor bytes out of the GGUF into one file (for fast local random reads).
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
t = [x for x in r.tensors if x.name == "per_layer_token_embd.weight"][0]
print(t.name, t.tensor_type.name, t.shape, t.n_bytes, "offset", t.data_offset)
with open(sys.argv[1], "rb") as f, open(sys.argv[2], "wb") as o:
    f.seek(t.data_offset); left = int(t.n_bytes)
    while left:
        b = f.read(min(left, 1 << 28)); o.write(b); left -= len(b)
print("done")
