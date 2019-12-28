python controller.py -n pairwise_lstm -config vox2_speakers_5994v118_LSTM512 -train
python controller.py -n pairwise_lstm -config vox2_speakers_5994v118_LSTM1024 -train

srun --pty --ntasks=1 --cpus-per-task=3 --mem=32G --gres=gpu:1 singularity exec ~/docker/zhaw_deep_voice-latest.simg python controller.py -n pairwise_lstm -config vox2_speakers_5994v118_LSTM512 -train

srun --pty --ntasks=1 --cpus-per-task=3 --mem=32G --gres=gpu:1 singularity exec ~/docker/zhaw_deep_voice-latest.simg python controller.py -n pairwise_lstm -config vox2_speakers_5994v118_LSTM1024 -train

srun --pty --ntasks=1 --cpus-per-task=3 --mem=32G --gres=gpu:1 singularity shell ~/docker/zhaw_deep_voice-latest.simg