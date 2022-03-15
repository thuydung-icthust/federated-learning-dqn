from torchvision import transforms, datasets
from utils.utils import *
from utils.gendata import *

def split_dataset(train_dataset, valid_percentage = 0.2):
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(
    train_dataset, [train_set_size, valid_set_size])
    path_to_file_json, path_to_valid_file_json = get_data_index(train_dataset, valid_set)
    return path_to_file_json, path_to_valid_file_json
    
    # list_idx_sample = load_dataset_idx(path_to_file_json)
    
    
    
    

def gen_dataset(dataset_name, path_data_idx, path_data_valid_idx = None):
    if dataset_name == "mnist":
        transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST("data/mnist/", train=True, download=True, transform=transforms_mnist)
        test_dataset = datasets.MNIST("data/mnist/", train=False, download=True, transform=transforms_mnist)
        # list_idx_sample = load_dataset_idx(path_data_idx)
        split_dataset(train_dataset)

    elif dataset_name == "cifar100":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10("./data/cifar/", train=True, download=True,transform=apply_transform)
        test_dataset = datasets.CIFAR10("./data/cifar/", train=False, download=True,transform=apply_transform)
        split_dataset(train_dataset)

        # list_idx_sample = load_dataset_idx(path_data_idx)
    elif dataset_name == "fashionmnist":
        transforms_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.FashionMNIST("./data/fashionmnist/", train=True, download=True,
                                         transform=transforms_mnist)
        test_dataset = datasets.FashionMNIST("./data/fashionmnist/", train=False, download=True,
                                        transform=transforms_mnist)
        split_dataset(train_dataset)
        
        # train_set_size = int(len(train_dataset) * 0.8)
        # valid_set_size = len(train_dataset) - train_set_size
        # train_set, valid_set = torch.utils.data.random_split(
        #     train_dataset, [train_set_size, valid_set_size])
        # path_to_file_json = get_data_index(train_dataset, train_set)
        
        # list_idx = json.load(open(path_data_valid_idx, 'r'))
        # list_idx = list(list_idx)
        # if path_data_valid_idx:
        #     valid_set = datasets.Subset(train_dataset, list_idx)
        #     list_idx_sample = load_dataset_idx(path_data_idx) 
        # else:
        #     list_idx_sample = load_dataset_idx(path_to_file_json)
        
    else:
        warnings.warn("Dataset not supported")
        exit()

    # return train_dataset, test_dataset, list_idx_sample, valid_set

if __name__ == '__main__':
    gen_dataset("fashionmnist")