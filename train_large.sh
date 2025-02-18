export XDG_CACHE_HOME=/disk/data4/zbrunner/.cache	# ensure cache is not on AFS
deactivate 2>/dev/null || true && source /disk/data4/zbrunner/whisper_biasing/espnet/tools/activate_python.sh
export CUDA_VISIBLE_DEVICES=0,1 
expdir=finetune_large_librispeech_lr0.0005_KB100_drop0.3
mkdir -p exp/${expdir}
python train.py \
    --modeltype large \
    --train_json data/LibriSpeech/train_clean_100_error.json \
    --dev_json data/LibriSpeech/dev_error.json \
    --lr 0.0005 \
    --batch_size 1 \
    --log_interval 200 \
    --nepochs 30 \
    --warmup_pct 0.0 \
    --decay_pct 0.2 \
    --expdir exp/${expdir} \
    --logfile exp/${expdir}/log.txt \
    --accumgrad 10 \
    --biasing \
    --biasinglist data/LibriSpeech/Blist/rareword_error.txt \
    --dropentry 0.3 \
    --maxKBlen 100 \
    # --loadfrom exp/${expdir}/model.acc.best \
