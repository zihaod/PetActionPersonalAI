import subprocess
import os
import argparse
from utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Cat Action Recognition Training Script')
    
    # Add path arguments
    parser.add_argument('--base_data_dir', 
                        type=str, 
                        default='/content/drive/MyDrive/pet_project/pet_action/6月10日json合集',
                        help='Directory containing base training data')
    
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='/content/drive/MyDrive/pet_project/pet_action/personal_AI_test_data/20250109数据标注/',
                        help='Directory containing personal test data')
    
    parser.add_argument('--pretrained_model_path',
                        type=str,
                        default='/content/drive/MyDrive/pet_project/pet_action/cat_models/cat_hidden_72_seq_20_scaledown_100_accdiff_20240622.pt',
                        help='Path to pretrained model weights')
    
    # Add optional training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=512,
                        help='Validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--date', type=str, default='20250109',
                        help='Date string for model naming')
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Define action labels
    actions = ['walk', 'sleep', 'run', 'lick', 'play', 'jump', 'feed', 'roll', 'scratch', 'rest']

    # class data multiplier
    idx2multiplier = dict()
    idx2multiplier[0] = 1
    idx2multiplier[1] = 1
    idx2multiplier[2] = 8
    idx2multiplier[3] = 1
    idx2multiplier[4] = 7
    idx2multiplier[5] = 10
    idx2multiplier[6] = 1
    idx2multiplier[7] = 15
    idx2multiplier[8] = 20
    idx2multiplier[9] = 1

    # Training hyperparams (now from args)
    num_epoch = args.num_epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    lr = args.learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    date = args.date

    # For dataset (now from args)
    base_data = []
    fpaths_dict = {}
    base_data_dir = args.base_data_dir
    data_dir = args.data_dir
    
    # Model weight ckpt (now from args)
    pretrained_model_path = args.pretrained_model_path
    
    #######################################

    # Load data from main database
    for fname in os.listdir(base_data_dir):
        if fname.startswith('猫') and fname.endswith('.json'):  # Using 猫 for cats
            fpath = os.path.join(base_data_dir, fname)
            data = read_json_data(fpath)
            base_data.append(data)
            

    # Group personal data files by identities
    fpaths = get_json_files(data_dir)
    keys = set(["_".join(p.split("/")[-1].split("_")[1:4]) for p in fpaths])
    for k in keys:
        fpaths_dict[k] = []
        for p in fpaths:
            if k in p:
                fpaths_dict[k].append(p)
    
    print(fpaths_dict)

    
    #######################################
    
    # Load base dataset
    dataset = ActionDataset(base_data, actions, seq_len=20, step_size=1, diff_mode="acc", augment=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get Base data labels and split based on label
    base_labels = set([dataset[i][1] for i in range(len(dataset))])
    base_split_data = {}
    for label in base_labels:
        base_split_data[label] = []
    
    for d in dataset:
        label = d[1]
        base_split_data[label].append(d)
        
    # Iterate through each identity
    for k, fpaths in fpaths_dict.items():
        if '猫' not in k:  # Using 猫 for cats
            continue

        print(k)
        personal_data = [read_json_data(fpath) for fpath in fpaths]
    
        # Load personal dataset
        personal_dataset = ActionDataset(personal_data, actions, seq_len=20, step_size=1, diff_mode="acc", augment=None)
        personal_loader = DataLoader(personal_dataset, batch_size=val_batch_size, shuffle=False)
    
        # Get personal data labels and split based on label
        personal_labels = set([personal_dataset[i][1] for i in range(len(personal_dataset))])
        personal_split_data = {}
        for label in personal_labels:
            personal_split_data[label] = []
    
        for d in personal_dataset:
            label = d[1]
            personal_split_data[label].append(d)
    
        # Construct combined dataset
        target_label_idxs = personal_labels
        combined_data = []
    
        for label in base_labels:
    
            if label in target_label_idxs:
                personal_len = len(personal_split_data[label])
                base_len = len(base_split_data[label])
    
                for _ in range(idx2multiplier[label]):
    
                    for _ in range(max(base_len // personal_len, 1)):
                        combined_data.extend(personal_split_data[label])
    
                    combined_data.extend(base_split_data[label])
    
                print(len(personal_split_data[label]))
                print(len(base_split_data[label]))
    
            else:
                combined_data.extend(base_split_data[label])
    
        combined_dataset = ActionDatasetFromGroup(combined_data, actions)
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        # Load pretrained model weight
        model = torch.load(pretrained_model_path).to(device)
        
        # Start training
        try:
            train(model, combined_loader, personal_loader, num_epoch, batch_size, lr, device)
            predictions, labels = evaluate(model, personal_loader, device=device)
            predictions, labels = evaluate(model, data_loader, device=device)
    
            model.cpu()
            model_name = 'train_' + k + '_' + date
            
            # Save model in binary format and zip the files
            export_lstm_to_bin(model_name, model, 72)
            zip_cmd = ["zip", "-r", model_name + ".zip", model_name]
    
            subprocess.run(zip_cmd)
    
        except:
            print(f"Failed file {fname}")
