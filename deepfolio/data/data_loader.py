import tensorflow as tf

'''
class PortfolioDataset(Dataset):
    def __init__(self, features, returns):
        self.features = features
        self.returns = returns
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.returns[idx]

def get_data_loader(features, returns, batch_size=32):
    dataset = PortfolioDataset(features, returns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
'''


def get_data_loader(features, returns, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((features, returns))
    return dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)