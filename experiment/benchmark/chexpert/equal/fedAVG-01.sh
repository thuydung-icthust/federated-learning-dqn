python3 -u train.py --train_mode="benchmark" --dataset_name="chexpert" --num_clients=10 --clients_per_round=10 --num_rounds=1000 --learning_rate=0.01 --batch_size=10 --num_epochs=5 --path_data_idx="dataset_idx/chexpert/CheXpert-noniid-mnist_equal_sample_1.json" --run_name="FedAVG-Equal-Non-IID-01-2.1.1" --group_name="CHEXPERT-FedAvg-Equal"