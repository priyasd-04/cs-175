import datasets
from sklearn.model_selection import train_test_split


def load_old_english_dataset(val_size:float = 0.1, test_size:float = 0.1,  /, seed:int|None = None):
    """
    test_size: Test size split [0-1.0]
    val_size: Val size split [0-1.0]
    seed: randomstate seed 

    Returns: (train_dataset, validation_dataset, test_dataset)
    """
    assert (test_size>=0 and val_size>=0 and test_size+val_size<=1)
    ds = datasets.load_dataset("apssg96/the-old-english-dataset",
                                split=datasets.Split.TRAIN, #Dataset only has one split: "train"
                              )['translation']
    if test_size + val_size == 0:
        return ds 
    train_ds, _ds  = train_test_split(ds, test_size=(test_size + val_size), train_size=1-(test_size + val_size),
                                      shuffle=True, random_state=seed)
    if val_size == 0:
        return train_ds, _ds
    val_ds, test_ds = train_test_split(_ds, test_size=test_size/(test_size+val_size), train_size=val_size/(test_size+val_size),
                                        shuffle=True, random_state=seed)

    return train_ds, val_ds, test_ds

def load_shakespeare_dataset(val_size:float = 0.1, test_size:float = 0.1,  /, seed:int|None = None):
    """
    test_size: Test size split [0-1.0]
    val_size: Val size split [0-1.0]
    seed: randomstate seed 

    Returns: (train_dataset, validation_dataset, test_dataset)
    """
    assert (test_size>=0 and val_size>=0 and test_size+val_size<=1)
    ds = datasets.load_dataset("Trelis/tiny-shakespeare",
                                split=datasets.Split.ALL, #Dataset only has one split: "train"
                              )['Text']
    if test_size + val_size == 0:
        return ds 
    train_ds, _ds  = train_test_split(ds, test_size=(test_size + val_size), train_size=1-(test_size + val_size),
                                      shuffle=True, random_state=seed)
    if val_size == 0:
        return train_ds, _ds
    val_ds, test_ds = train_test_split(_ds, test_size=test_size/(test_size+val_size), train_size=val_size/(test_size+val_size),
                                        shuffle=True, random_state=seed)

    return train_ds, val_ds, test_ds

#train_ds = load_old_english_dataset(0.0, 0.0)
#print("Train: ", len(train_ds), "Val: ", len(val_ds), "Test ", len(test_ds))
#text_lengths = [(len(x)) for x in train_ds]
#print(len(train_ds))
#print(min(text_lengths)) #34
#print(max(text_lengths)) #1494

#sp = load_shakespeare_dataset(0.0,0.0)
#print(len(sp))
