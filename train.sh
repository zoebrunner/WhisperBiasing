export XDG_CACHE_HOME=/disk/data4/zbrunner/.cache       # ensure cache is not on AFS
deactivate 2>/dev/null || true && source /disk/data4/zbrunner/whisper_biasing/espnet/tools/activate_python.sh
export CUDA_VISIBLE_DEVICES=0,1 
mkdir -p exp/${expdir}
python train.py \
    --train_json data/LibriSpeech/train_clean_100_f15.json \
    --dev_json data/LibriSpeech/dev_f15.json \
    --lr 0.0005 \
    --batch_size 16 \
    --log_interval 20 \
    --nepochs 30 \
    --warmup_pct 0.0 \
    --decay_pct 0.2 \
    --expdir exp/${expdir} \
    --logfile exp/${expdir}/log.txt \
    --accumgrad 5 \
    --biasing \
    --biasinglist data/LibriSpeech/Blist/rareword_f15.txt \
    --dropentry 0.1 \
    --maxKBlen 200 \
    # --useGPT \
    # --GNNtype gcn2 \
    # --GNNdim 256 \
    # --loadfrom exp/finetune_librispeech_lr0.0005_KB200_drop0.1/model.acc.best \
