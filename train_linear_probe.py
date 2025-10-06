import argparse, time, json
from pathlib import Path

import torch, torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets as D
from sklearn.metrics import accuracy_score

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

def get_model(name, num_classes):
    name=name.lower()
    if name=="resnet50":
        m=torchvision.models.resnet50(weights="IMAGENET1K_V2")
        in_f=m.fc.in_features
        m.fc=nn.Linear(in_f, num_classes)
        backbone_params = [p for n,p in m.named_parameters() if not n.startswith("fc.")]
    elif name in ["efficientnet_b0","efficientnet-b0","effnet_b0"]:
        m=torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_f=m.classifier[1].in_features
        m.classifier[1]=nn.Linear(in_f, num_classes)
        backbone_params = [p for n,p in m.named_parameters() if not n.startswith("classifier.1")]
    else:
        raise ValueError("resnet50 | efficientnet_b0")
    # freeze backbone
    for p in backbone_params: p.requires_grad=False
    return m

def get_loaders(data_root, img_size=224, batch=64, workers=4):
    train_tfm=T.Compose([
        T.Resize(256), T.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    eval_tfm=T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_ds=D.ImageFolder(Path(data_root)/"train", transform=train_tfm)
    val_ds  =D.ImageFolder(Path(data_root)/"val",   transform=eval_tfm)
    train_dl=DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True)
    val_dl  =DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ds, val_ds, train_dl, val_dl

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    ys=[]; yh=[]
    for x,y in dl:
        x=x.to(device,non_blocking=True)
        logits=model(x)
        yh.append(logits.argmax(1).cpu())
        ys.append(y)
    yh=torch.cat(yh); ys=torch.cat(ys)
    return float(accuracy_score(ys.numpy(), yh.numpy()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_prepared")   # 5-class fruit (top-level folders)
    ap.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0","resnet50"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--outdir", default="checkpoints")
    args=ap.parse_args()

    train_ds, val_ds, train_dl, val_dl = get_loaders(args.data, args.img_size, args.batch, args.workers)
    num_classes=len(train_ds.classes)
    device="cuda" if torch.cuda.is_available() else "cpu"

    model=get_model(args.model, num_classes).to(device)
    # optimize only the head
    head_params=[p for p in model.parameters() if p.requires_grad]
    opt=optim.AdamW(head_params, lr=args.lr)
    crit=nn.CrossEntropyLoss()

    best=0.0
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path=outdir/f"{args.model}_linear_probe_best.pth"

    for ep in range(1, args.epochs+1):
        model.train()
        t0=time.time(); loss_sum=0; n=0
        for x,y in train_dl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad()
            logits=model(x)
            loss=crit(logits,y)
            loss.backward(); opt.step()
            loss_sum+=loss.item()*y.size(0); n+=y.size(0)
        train_loss=loss_sum/max(n,1)
        val_acc=evaluate(model, val_dl, device)
        dt=time.time()-t0
        print(f"Epoch {ep}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)")
        if val_acc>best:
            best=val_acc
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, ckpt_path)
    with open(outdir/f"{args.model}_linear_probe_summary.json","w") as f:
        json.dump({"best_val_acc":best, "classes":train_ds.classes}, f, indent=2)
    print(f"Best val_acc={best:.4f}  -> {ckpt_path}")

if __name__=="__main__":
    main()
