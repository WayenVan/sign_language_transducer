import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf
from csi_sign_language.models.model_rna import SLRTransducer

def test_hrnet_rnn():
    cfg = "configs/train/hrnet_rnn.yaml"
    cfg = OmegaConf.load(cfg)
    loader = instantiate(cfg.data.train_loader)
    model = instantiate(cfg.model, vocab=loader.dataset.get_vocab()).to(cfg.device)
    data = next(iter(loader))
    video = data['video'].to(cfg.device)
    lgt = data['video_length'].to(cfg.device)

    output = model(video, lgt)
    return

def test_transducer():
    cfg = "configs/train/transducer.yaml"
    cfg = OmegaConf.load(cfg)
    loader = instantiate(cfg.data.train_loader)
    model = instantiate(cfg.model, vocab=loader.dataset.get_vocab()).to(cfg.device)
    data = next(iter(loader))
    video = data['video'].to(cfg.device)
    lgt = data['video_length'].to(cfg.device)
    target = data['gloss'].to(cfg.device)
    target_lgt = data['gloss_length'].to(cfg.device)
    output = model(video, lgt, target, target_lgt)
    loss = model.criterion(output, target, target_lgt)
    loss.backward()
    labels = model.inference(video, lgt)
    print(labels)
    return


if __name__ == "__main__":
    test_transducer()
    
    