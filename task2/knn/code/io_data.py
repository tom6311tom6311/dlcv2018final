import os, sys
from skimage import color, io
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift

#np.random.seed(4208387457)
print('io', np.random.get_state()[1][0])

def read_imgs(path):
    #img = io.imread(path, as_grey=True)
    img = io.imread(path)
    img = np.expand_dims(np.array(img), axis = 0)
    if len(img.shape)<4:
        img = np.expand_dims(np.array(img), axis = -1)
    return img

def img_preprocessing(imgs):
    #imgs = np.vstack(imgs).astype(np.float32) / 128 - 1
    imgs = np.vstack(imgs).astype(np.float32) / 256
    return imgs

def data_augmentation(imgs, label):
    print(imgs.shape)
    imgs = np.vstack([imgs, 
        shift(imgs, [0, 0.1, 0, 0], mode = 'nearest'),
        shift(imgs, [0, -0.1, 0, 0], mode = 'nearest'),
        shift(imgs, [0, 0, 0.1,  0], mode = 'nearest'),
        shift(imgs, [0, 0, -0.1, 0], mode = 'nearest'),
        shift(imgs, [0, 0.05, 0.05, 0], mode = 'nearest'),
        shift(imgs, [0, -0.05, 0.05, 0], mode = 'nearest'),
        shift(imgs, [0, 0.05, -0.05,  0], mode = 'nearest'),
        shift(imgs, [0, -0.05, -0.05, 0], mode = 'nearest')])
    label = np.reshape(np.vstack([label]*9), (-1))

    imgs = np.vstack([imgs, np.flip(imgs, axis=2)])
    label = np.reshape(np.vstack([label]*2), (-1))
    return imgs, label

def read_pretrain(path, base=True, sample=5):
    '''
    if m == True: return sat_name, sat, mask
    else: return sat_name, sat, []
    '''

    base_path = os.path.join(path, 'base')
    
    train_imgs = []
    train_label = []
    test_imgs = []
    test_label = []

    if base:
        base_file = sorted([i for i in os.listdir(base_path)])
    
        for f in base_file:
            imgs_name = sorted([i for i in os.listdir(os.path.join(base_path, f, 'train'))])
            for i in imgs_name:
                img = read_imgs(os.path.join(base_path, f, 'train', i))
                train_imgs.append(img)
                train_label.append(f[-2:])

        for f in base_file:
            imgs_name = sorted([i for i in os.listdir(os.path.join(base_path, f, 'test'))])
            for i in imgs_name:
                img = read_imgs(os.path.join(base_path, f, 'test', i))
                test_imgs.append(img)
                test_label.append(f[-2:])

    train_imgs = img_preprocessing(train_imgs)
    train_label = np.array(train_label)
    test_imgs = img_preprocessing(test_imgs)
    test_label = np.array(test_label)
    

    return train_imgs, train_label, test_imgs, test_label

def read_train(path, base=True, sample=5):
    '''
    if m == True: return sat_name, sat, mask
    else: return sat_name, sat, []
    '''

    base_path = os.path.join(path, 'base')
    novel_path = os.path.join(path, 'novel')
    novel_file = sorted([i for i in os.listdir(novel_path)])
    
    novel_imgs_name = []
    novel_imgs = []
    novel_label = []
    for f in novel_file:
        imgs_name = sorted([i for i in os.listdir(os.path.join(novel_path, f, 'train'))])
        imgs_name = np.random.choice(imgs_name, size=sample, replace=False)
        for i in imgs_name:
            img = read_imgs(os.path.join(novel_path, f, 'train', i))
            novel_imgs_name.append(os.path.join(f, 'train', i))
            novel_imgs.append(img)
            novel_label.append(f[-2:])

    with open('novel_imgs_name' + str(sample) + '.txt', 'w') as output:
        output.write("\n".join(novel_imgs_name))

    base_imgs = []
    base_label = []
    if base:
        base_path = os.path.join(path, 'base')
        base_file = sorted([i for i in os.listdir(base_path)])
    
        for f in base_file:
            imgs_name = sorted([i for i in os.listdir(os.path.join(base_path, f, 'train'))])
            for i in imgs_name:
                img = read_imgs(os.path.join(base_path, f, 'train', i))
                base_imgs.append(img)
                base_label.append(f[-2:])


    novel_imgs = img_preprocessing(novel_imgs)
    novel_label = np.array(novel_label)
    base_imgs = img_preprocessing(base_imgs)
    base_label = np.array(base_label)
    
    novel_imgs, novel_label = data_augmentation(novel_imgs, novel_label)

    return novel_imgs, novel_label, base_imgs, base_label


def read_test(train_path, test_path, random=False, sample=5):
    ## Read real test data #

    novel_imgs_name = []
    train_imgs = []
    test_imgs = []
    label = []

    if not random:
        with open('novel_imgs_name' + str(sample) + '.txt', 'r') as ifile:
            novel_path = ifile.read().splitlines()
        for f in novel_path:
            img = read_imgs(os.path.join(train_path, 'novel', f))
            train_imgs.append(img)
            l = f.find('_')+1
            label.append(f[l:l+2])
    else:
        novel_path = os.path.join(train_path, 'novel')
        novel_file = sorted([i for i in os.listdir(novel_path)])
        
        for f in novel_file:
            imgs_name = sorted([i for i in os.listdir(os.path.join(novel_path, f, 'train'))])
            imgs_name = np.random.choice(imgs_name, size=sample, replace=False)
            for i in imgs_name:
                img = read_imgs(os.path.join(novel_path, f, 'train', i))
                novel_imgs_name.append(os.path.join(f, 'train', i))
                train_imgs.append(img)
                label.append(f[-2:])

        with open('test_novel_imgs_name' + str(sample) + '.txt', 'w') as output:
            output.write("\n".join(novel_imgs_name))
        


    test_file = sorted([i for i in os.listdir(test_path)])
    test_idx = [int(i[:i.find('.')]) for i in test_file]
    test_idx, test_file = zip(*sorted(zip(test_idx, test_file)))
    for f in test_file:
        img = read_imgs(os.path.join(test_path, f))
        test_imgs.append(img)

    train_imgs = img_preprocessing(train_imgs)
    test_imgs = img_preprocessing(test_imgs)
    label = np.array(label)

    return train_imgs, label, test_imgs

def save_predict(predict, name):
    ## save predict 
    idx = np.arange(len(predict))
    predict = list(map(int, predict))
    df = pd.DataFrame({'image_id':idx, 'predicted_label':predict})
    df.to_csv(name, index=False)

