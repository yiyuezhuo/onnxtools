# ONNX manipulating tools

I want to manipulate existing ONNX for adding some pre/post-processing logic so that I can port model between language more easily. Unfortunately I found it's very annoy, so I write this tool to simplify coding somewhat.

## Usage

Add some preprocess and post process logic into a ONNX model exported from PyTorch:

```python

model = onnx.load("resnet18.onnx")

remove_input_head(model)

with namespace("preprocess", extern=["input_img_seq", "input"]), env(model, append=False):
    input("input_img_seq", TensorProto.UINT8, shape=["batch_size", 224, 224, 3])
    
    node("Cast", ["input_img_seq"], ["input_non_centered_scaled_255"], to=TensorProto.FLOAT)
    node("Div", ["input_non_centered_scaled_255", 255.], ["input_non_centered_scaled"])
    node("Sub", ["input_non_centered_scaled", [0.485, 0.456, 0.406]], ["input_non_scaled"])
    node("Div", ["input_non_scaled", [0.229, 0.224, 0.225]], ["input_bwhc"])
    node("Transpose", ['input_bwhc'], ["input"], perm=[0, 3, 1, 2])


with namespace("postprocess", extern=["output", "indices_1", "probs_top1_1"]), env(model):
    node("TopK", ["output", 1], ['values', 'indices'])
    node("Softmax", ['output'], ['probs'])
    node("GatherElements", ["probs", "indices"], ['probs_top1'], axis=1)
    node("Squeeze", ["probs_top1", 1], ['probs_top1_1'])
    node("Squeeze", ["indices", 1], ['indices_1'])
    
    output("indices_1", TensorProto.INT64, shape=["batch_size"])
    output("probs_top1_1", TensorProto.FLOAT, shape=["batch_size"])

onnx.checker.check_model(model, full_check=True)

```

Try it:

```python
import torch
from torchvision import transforms
from PIL import Image

resize_size, crop_size = 256, 224

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

test_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

pure_image_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
])

def preprocess(img_list):
    return torch.stack([test_transform(img) for img in img_list], dim=0)

def postprocess(output):
    # If we just want what the black-box guy usually want (and only "know"), classidx and somewhat "score"
    indices = output.topk(dim=1, k=1).indices[:, 0] # while `argsort` is more suit here, topk is used to match onnx interface.
    probs = F.softmax(output, dim=1)
    return indices, probs[torch.arange(len(indices)), indices]

hp = Image.open("hp.jpg")
batch = preprocess([hp, hp])
batch_py = batch.numpy()
output_org = session_org.run(["output"], {"input": batch_py})[0]
"""
array([[-1.9975376 , -0.4984647 ,  1.3288195 , ..., -0.06410396,
        -3.5900416 ,  0.20868209],
       [-1.9975376 , -0.4984647 ,  1.3288195 , ..., -0.06410396,
        -3.5900416 ,  0.20868209]], dtype=float32)
"""

batch_tool = np.stack([np.array(pure_image_transform(hp)), np.array(pure_image_transform(hp))], axis=0)
output_tool = session_tool.run(None, {"input_img_seq": batch_tool})
"""
[array([[-1.9975376 , -0.4984647 ,  1.3288195 , ..., -0.06410396,
         -3.5900416 ,  0.20868209],
        [-1.9975376 , -0.4984647 ,  1.3288195 , ..., -0.06410396,
         -3.5900416 ,  0.20868209]], dtype=float32),
 array([917, 917], dtype=int64),
 array([0.94007367, 0.94007367], dtype=float32)]
"""
```

## References

* https://github.com/onnx/onnx/blob/master/docs/IR.md
* https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
* https://github.com/onnx/onnx/blob/master/docs/Operators.md
* https://github.com/onnx/onnx/issues/2216#issuecomment-667760885

