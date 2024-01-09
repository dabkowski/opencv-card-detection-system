import torch
from fastai.vision.all import *

print(torch.cuda.is_available())

# Dataloader
cards = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
).dataloaders('data', bs=32)

# learner
learn = vision_learner(cards, resnet50, metrics=error_rate)

# Train model 20 epochs
learn.fine_tune(20)

learn.export('card_classifier_update.pkl')