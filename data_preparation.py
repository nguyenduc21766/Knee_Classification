import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

metadata = pd.DataFrame(columns=["Name", "Path","KL"])

main_dir = r"data/OSAIL_KL_Dataset/Labeled"

for grade in range(5):
    folder_path = os.path.join(main_dir, str(grade))

    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)

        metadata = metadata._append({
            "Name": image,
            "Path": image_path,
            "KL": grade
        }, ignore_index=True)

metadata.to_csv("metadata.csv", index=False)



'''
# Visualize KL distribution for entire dataset
metadata['KL'].value_counts().sort_index().plot(kind='bar')
plt.title("KL Grade Distribution - Entire Dataset")
plt.xlabel("KL Grade")
plt.ylabel("Count")
plt.savefig("KL_distribution_entire.png")
plt.close()
'''
metadata['KL'] = metadata['KL'].astype(int)
train_val, test = train_test_split(metadata, test_size=0.2, stratify=metadata['KL'], random_state=42)
#train, val = train_test_split(train_val, test_size=0.2, stratify=train_val['KL'], random_state=42)

'''
for df, name in zip([train, val, test], ["Train", "Validation", "Test"]):
    df['KL'].value_counts().sort_index().plot(kind='bar')
    plt.title(f"KL Grade Distribution - {name} Set")
    plt.xlabel("KL Grade")
    plt.ylabel("Count")
    plt.savefig(f"KL_distribution_{name.lower()}.png")
    plt.close()
'''




#  Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, train_val['KL']), 1):
    train_fold = train_val.iloc[train_idx]
    val_fold = train_val.iloc[val_idx]

    #  Plot KL distribution for training fold
    train_fold['KL'].value_counts().sort_index().plot(kind='bar')
    plt.title(f"KL Distribution - Train Fold {fold}")
    plt.xlabel("KL Grade")
    plt.ylabel("Count")
    plt.savefig(f"KL_distribution_train_fold{fold}.png")
    plt.close()

    # Plot KL distribution for validation fold
    val_fold['KL'].value_counts().sort_index().plot(kind='bar')
    plt.title(f"KL Distribution - Validation Fold {fold}")
    plt.xlabel("KL Grade")
    plt.ylabel("Count")
    plt.savefig(f"KL_distribution_val_fold{fold}.png")
    plt.close()

# Step 5: Optional - Plot KL distribution for test set
test['KL'].value_counts().sort_index().plot(kind='bar')
plt.title("KL Distribution - Test Set")
plt.xlabel("KL Grade")
plt.ylabel("Count")
plt.savefig("KL_distribution_test.png")
plt.close()