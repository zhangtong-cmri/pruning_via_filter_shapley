import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def get_loader_cifar10(args, kwargs=None):
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]
    loader_train = None

    if kwargs == None:
        kwargs = {
            'num_workers': args.n_threads,
            'pin_memory': True
        }
    
    if args.is_train:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(0, transforms.RandomHorizontalFlip())
        
        transform_train = transforms.Compose(transform_list)

        loader_train = DataLoader(
            datasets.CIFAR10(
                root=args.dir_data,
                train=True,
                download=True,
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.CIFAR10(
            root=args.dir_data,
            train=False,
            download=True,
            transform=transform_test),
        batch_size=500, shuffle=False)

    return loader_train, loader_test
