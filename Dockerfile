FROM nvcr.io/nvidia/pytorch:21.02-py3
LABEL org.opencontainers.image.authors="h.kuijf@umcutrecht.nl"

RUN mkdir -p /home

WORKDIR /home

RUN python -m pip install -U pip


COPY requirements.txt /home/
RUN python -m pip install -r requirements.txt

COPY process.py MCB_3DUNet_high.pt MCB_3DUNet_low.pt MCB_MaskRCNN_high.pt MCB_MaskRCNN_low.pt MCBS_predictions.py maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth /home/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=mixmicrobleed

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=10G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=10G