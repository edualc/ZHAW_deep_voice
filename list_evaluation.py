import h5py

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

utterances = dict()
speakers = dict()

for key in eval_lists:
    utterances[key] = list()
    speakers[key] = list()

    for line in open(eval_lists[key],'r'):
        label, file1, file2 = line[:-1].split(' ')

        # import code; code.interact(local=dict(globals(), **locals()))

        utterances[key].append(file1)
        utterances[key].append(file2)

        speakers[key].append(file1.split('/')[0])
        speakers[key].append(file2.split('/')[0])

    utterances[key] = set(utterances[key])
    speakers[key] = set(speakers[key])

import code; code.interact(local=dict(globals(), **locals()))

for key in utterances:
    print(key, len(speakers[key]), len(utterances[key]))