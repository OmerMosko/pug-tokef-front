import torch
from detection_project.models.crnn import CRNN
model = CRNN(32, 1, 37, 256)
model.load_state_dict(torch.load('netCRNN_24_50.pth'))
dummy_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, dummy_input, "netCRNN_24_50.onnx", verbose=True)