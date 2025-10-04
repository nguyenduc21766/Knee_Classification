from args import get_args
import pandas as pd
import os
from datasets import Knee_Xray_dataset
import torch
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model

def main():
    # 1. need some arguments
    args = get_args()
    print("Args received:", args)
    # 2. Iterate among the folds
    for fold in range(1,6):
        print('Training fold:', fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_train.csv'.format(fold)))
        val_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_val.csv'.format(fold)))

        # 3. Preparing dataset
        train_dataset = Knee_Xray_dataset(train_set)
        val_dataset = Knee_Xray_dataset(val_set)

        # 4. Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


        # 5. Initializing the model
        model = MyModel(args.backbone)

        # 6. Train the model
        train_model(model, train_loader, val_loader)




    print()


if __name__ == '__main__':
    main()