import subprocess



if __name__ == '__main__':

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

    # Load data from main database
    cat_data = []
    for fname in os.listdir('/content/drive/MyDrive/pet_project/pet_action/6月10日json合集'):
        if fname.startswith('猫') and fname.endswith('.json'):
            fpath = '/content/drive/MyDrive/pet_project/pet_action/6月10日json合集/' + fname
            data = read_json_data(fpath)
            cat_data.append(data)
    
    # Define action labels
    actions = ['walk', 'sleep', 'run', 'lick', 'play', 'jump', 'feed', 'roll', 'scratch', 'rest']
    
    # Load base dataset
    dataset = ActionDataset(cat_data, actions, seq_len=20, step_size=1, diff_mode="acc", augment=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get Base data labels and split based on label
    base_labels = set([dataset[i][1] for i in range(len(dataset))])
    base_split_data = {}
    for label in base_labels:
        base_split_data[label] = []
    
    for d in dataset:
        label = d[1]
        base_split_data[label].append(d)
    
    

    for k, fpaths in fpaths_dict.items():
        if '猫' not in k:
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
        model = torch.load('/content/drive/MyDrive/pet_project/pet_action/cat_models/cat_hidden_72_seq_20_scaledown_100_accdiff_20240622.pt').to(device)

        # Start training
        try:
          train(model, combined_loader, personal_loader, num_epoch, batch_size, lr, device)
          predictions, labels = evaluate(model, personal_loader, device=device)
          predictions, labels = evaluate(model, data_loader, device=device)
    
          model.cpu()
          model_name = 'train_' + k + '_' + date
          
          # Save model in binary format
          export_lstm_to_bin(model_name, model, 72)
          zip_cmd = ["zip", "-r", model_name + ".zip", model_name]
    
          subprocess.run(zip_cmd)
    
        except:
          print(f"Failed file {fname}")
