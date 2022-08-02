from tool2 import darknet2pytorch
import torch

# load weights from darknet format
model = darknet2pytorch.Darknet('checkpoints/yolov4.cfg', inference=True)
model.load_weights('checkpoints/yolov4.weights')

# save weights to pytorch format
torch.save(model.state_dict(), 'checkpoints/yolov4.pth')

# reload weights from pytorch format
model_pt = darknet2pytorch.Darknet('checkpoints/yolov4.cfg', inference=True)
model_pt.load_state_dict(torch.load('checkpoints/yolov4.pth'))