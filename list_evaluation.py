import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

datasets = {
    'vox1_dev': '/mnt/all1/voxceleb1/dev/vox1_dev_mel.h5',
    'vox1_test': '/mnt/all1/voxceleb1/test/vox1_test_mel.h5'
}

eval_lists = {
    'vox1': '/home/claude/development/ZHAW_deep_voice/common/data/evaluation_lists/list_vox1.txt',
    'vox1-cleaned': '/home/claude/development/ZHAW_deep_voice/common/data/evaluation_lists/list_vox1_c.txt',
    'vox1-E': '/home/claude/development/ZHAW_deep_voice/common/data/evaluation_lists/list_vox1_e.txt',
    'vox1-E-cleaned': '/home/claude/development/ZHAW_deep_voice/common/data/evaluation_lists/list_vox1_ec.txt',
    'vox1-H': '/home/claude/development/ZHAW_deep_voice/common/data/evaluation_lists/list_vox1_h.txt',
    'vox1-H-cleaned': '/home/claude/development/ZHAW_deep_voice/common/data/evaluation_lists/list_vox1_hc.txt',
}

# utterances = dict()
# speakers = dict()

# for key in eval_lists:
#     utterances[key] = list()
#     speakers[key] = list()

#     for line in open(eval_lists[key],'r'):
#         label, file1, file2 = line[:-1].split(' ')

#         # import code; code.interact(local=dict(globals(), **locals()))

#         utterances[key].append(file1)
#         utterances[key].append(file2)

#         speakers[key].append(file1.split('/')[0])
#         speakers[key].append(file2.split('/')[0])

#     utterances[key] = set(utterances[key])
#     speakers[key] = set(speakers[key])

# import code; code.interact(local=dict(globals(), **locals()))

# for key in utterances:
#     print(key, len(speakers[key]), len(utterances[key]))

meta_file_path = '/mnt/all1/voxceleb1/vox1_meta.csv'
data_file_path = '/mnt/all1/experiments/__VT1/A_vox2_arccos_eval_50percent_shift/pairwise_lstm_vox2_LSTM512_seg400_arc_cos_best.h5__eval_data.h5'
# pyplot_colors = [ 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan' ]

def get_color():
    return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

df = pd.read_csv(meta_file_path, delimiter='\t')
df = df.rename(columns={'VoxCeleb1 ID':'speaker', 'Gender':'sex', 'Nationality':'nationality'})
# df['speaker'] = df['speaker'].str.strip()
# df['sex'] = df['sex'].str.strip()
gender_check_arr = np.array(df[['speaker','sex','nationality']])

def gender_check(speaker):
    tmp = gender_check_arr[np.where(gender_check_arr[:,0] == speaker), 1][0]

    try:
        return tmp[0]
    except IndexError:
        return 'u'

def color_for_label(label):
    gender = gender_check(label.split('/')[0])

    if gender == 'm':
        return '#0000FF'
    elif gender == 'f':
        return '#FF69B4'
    else:
        return '#D3D3D3'

# import code; code.interact(local=dict(globals(), **locals()))

for gender in [True, False]:

    # for NUM_UTTERANCES in [8]:
    # for NUM_UTTERANCES in [8, 16, 32, 64, 128]:
    for NUM_UTTERANCES in [8192]:
        np.random.seed(621993)

        # NUM_UTTERANCES = 128
        pyplot_colors = [get_color() for i in range(NUM_UTTERANCES)]

        with h5py.File(data_file_path, 'r') as f:
            uniques, unique_counts  = np.unique(f['labels'][:], return_counts=True)
            
            labels = np.random.choice(uniques, NUM_UTTERANCES)
            all_labels = np.copy(f['labels'][np.isin(f['labels'][:],labels)])
            spectrograms = np.copy(f['embeddings'][np.isin(f['labels'][:], labels),:])

            # perform PCA
            # 
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(spectrograms)

            plt.figure(figsize=(10, 12))
            plt.subplot(2, 1, 1)
            plt.title('PCA of {} utterances'.format(NUM_UTTERANCES))

            for i, label in enumerate(labels):
                pca_specs = pca_result[np.where(all_labels == label), :][0, :, :]
                
                if gender:
                    plt.scatter(pca_specs[:,0], pca_specs[:,1], color=color_for_label(label), s=10)
                else:
                    plt.scatter(pca_specs[:,0], pca_specs[:,1], color=pyplot_colors[i], s=10)

            # # perform t-SNE
            # # 
            # tsne = TSNE(n_components=2, verbose=1, perplexity=NUM_UTTERANCES // 2, n_iter=2500)
            # tsne_results = tsne.fit_transform(spectrograms)

            plt.subplot(2, 1, 2)
            plt.title('t-SNE of {} utterances'.format(NUM_UTTERANCES))

            # for i, label in enumerate(labels):
            #     tsne_specs = tsne_results[np.where(all_labels == label), :][0, :, :]
                
            #     if gender:
            #         plt.scatter(tsne_specs[:,0], tsne_specs[:,1], color=color_for_label(label), s=10)
            #     else:
            #         plt.scatter(tsne_specs[:,0], tsne_specs[:,1], color=pyplot_colors[i], s=10)

            if gender:
                plt.savefig('plot_pca_tsne__' + str(NUM_UTTERANCES) + '_gender.png')
            else:
                plt.savefig('plot_pca_tsne__' + str(NUM_UTTERANCES) + '.png')
            plt.close()

            # import code; code.interact(local=dict(globals(), **locals()))
            