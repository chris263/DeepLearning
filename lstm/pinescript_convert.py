import torch, json

# Adjust to match your actual model class/load
model = torch.load("model.pt", map_location="cpu")
model.eval()

meta = json.load(open("meta.json"))
pre = json.load(open("preprocess.json"))

# Assuming single-layer LSTM named `lstm` and linear `fc`
lstm = model.lstm
fc = model.fc

state = model.state_dict()

w_ih = state["lstm.weight_ih_l0"].tolist()  # [4H, input_size]
w_hh = state["lstm.weight_hh_l0"].tolist()  # [4H, H]
b_ih = state["lstm.bias_ih_l0"].tolist()    # [4H]
b_hh = state["lstm.bias_hh_l0"].tolist()    # [4H]

w_fc = state["fc.weight"].tolist()          # [out_dim, H]
b_fc = state["fc.bias"].tolist()            # [out_dim]

# Example: export as Pine-ready text
def to_pine_array(name, values):
    flat = [f"{v:.10f}" for v in (sum(values, []) if any(isinstance(x, list) for x in values) else values)]
    return f"var float[] {name} = array.from({', '.join(flat)})"

print("// LSTM weights")
print(to_pine_array("W_IH", w_ih))
print(to_pine_array("W_HH", w_hh))
print(to_pine_array("B_IH", b_ih))
print(to_pine_array("B_HH", b_hh))

print("// FC weights")
print(to_pine_array("W_FC", w_fc))
print(to_pine_array("B_FC", b_fc))

# Also print preprocess params from `pre` as Pine arrays/consts
