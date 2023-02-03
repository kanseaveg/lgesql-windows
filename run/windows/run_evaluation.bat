set task=evaluation
set read_model_path=saved_models\%1
set batch_size=20
set beam_size=5
set device=0

python scripts/text2sql.py --task %task% --testing --read_model_path %read_model_path% ^
    --batch_size %batch_size% --beam_size %beam_size% --device %device%
