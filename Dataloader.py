'''
    实现Dataloader函数,用于加载数据集
    要求使用MyTensor的形式存储数据
'''
import numpy as np
import MyTensor


class Dataset:
    '''
    Dataset 的虚基类，需要被继承(子类需要实现__getitem__和__len__方法)
    '''

    def __init__(self) -> None:
        pass
    
    def __getitem__(self, idx):
        '''
        retrieve the idx-th data
        '''
        NotImplementedError
    
    def __len__(self):
        '''
        total number of data
        '''
        NotImplementedError

class Sampler:

    '''
    Sampler 的虚基类，需要被继承
    '''

    def __init__(self, dataset : Dataset) -> None:
        pass

    def __iter__(self):
        NotImplementedError

class RandomSampler(Sampler):
    '''
    随机采样器，从数据集中随机抽取样本，对应 shuffle = True 的情况
    '''

    def __init__(self, dataset : Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(np.random.permutation(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
class SequentialSampler(Sampler):
    '''
    顺序采样器，默认情况
    '''

    def __init__(self, dataset : Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
class BatchSampler(Sampler):
    '''
    批次采样器，将数据集分成多个批次
    '''

    def __init__(self, sampler: Sampler, batch_size:int, drop_last:bool)->None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler)//self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1)//self.batch_size
        
class Iterater:
    '''
    Dataloader 的迭代器类，用于迭代 Dataloader 的数据
    '''

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.iterater = iter(self.dataloader.batch_sampler)
    
    def __next__(self):
        idx = next(self.iterater)
        return self.dataloader.dataset[idx]


class DataLoader:
    '''
    DataLoader 类，用于加载数据集
    '''

    def __init__(self, dataset: Dataset, batch_size:int = 1, shuffle=False, drop_last: bool=False) -> None:
        '''
        初始化DataLoader
        input:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据集
            drop_last: 是否丢弃最后一个不足batch_size的批次
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle == True:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)


    def __iter__(self):
        return Iterater(self)



def mydataLoader(X, y, batch_size, shuffle=False, drop_last=False)->list:
    '''
    加载数据集
    input:
        X: 数据集特征
        y: 数据集标签
        batch_size: 批次大小
        shuffle: 是否打乱数据集
        drop_last: 是否丢弃最后一个不足batch_size的批次
    output:
        list: 返回一个列表，包含每个批次的数据
    '''
    class myDataset(Dataset):
        def __init__(self, X, y) -> None:
            self.data = X
            self.target = y

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx]
        
        def __len__(self):
            return len(self.data)
    
    return DataLoader(myDataset(X,y), batch_size, shuffle, drop_last)
    
if __name__ == '__main__':

    '''
    简单测试
    '''

    X = MyTensor.MyTensor(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]))
    y = MyTensor.MyTensor(np.array([[1,0],[2,0],[3,0],[4,0]]))
    dataloader = mydataLoader(X, y, batch_size=2, shuffle=False)
    # print(len(dataloader.dataset))
    # print(dataloader.dataset[0])
    for batch in dataloader:
        print(batch)

    X1 = MyTensor.MyTensor(np.array([1,2,3,4,5,6,7,8]))
    y1 = MyTensor.MyTensor(np.array([1,0,2,0,3,0,4,0]))
    dataloader1 = mydataLoader(X1, y1, batch_size=3, shuffle=True, drop_last=False)
    for batch in dataloader1:
        print(batch)
    
    # X1 = torch.tensor([[1,2,3,4],[5,6,7,8]])
    # y1 = torch.tensor([[1,0],[2,0]])
    # dataloader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X1, y1), batch_size=1, shuffle=False)
    # for batch in dataloader2:
    #     print(batch)