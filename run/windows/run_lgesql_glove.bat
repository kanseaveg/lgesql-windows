:: .\run\windows\run_lgesql_glove.bat [mmc|msde]

set task=lgesql_glove
set seed=999
set device=0
::'--testing'
set testing=
set read_model_path=

set model=lgesql
:: without_pruning  是否使用剪枝辅助工作以便提高encoder的识别能力
set output_model=with_pruning
:: mmc, msde, local   mmc:multi-head multi-view concatenation   msde: mixed static and dynamic embeddings 两种拆分头的方法
set local_and_nonlocal=%1
set embed_size=300
set schema_aggregation=head+tail
set gnn_hidden_size=256
set gnn_num_layers=8
set relation_share_heads=
set score_function=affine
set num_heads=8
set dropout=0.2
set attn_drop=0.0
set drop_connect=0.2

set lstm=onlstm
set chunk_size=8
set att_vec_size=512
set sep_cxt=
set lstm_hidden_size=512
set lstm_num_layers=1
set action_embed_size=128
set field_embed_size=64
set type_embed_size=64
set no_context_feeding=--no_context_feeding
set no_parent_production_embed=
set no_parent_field_embed=
set no_parent_field_type_embed=
set no_parent_state=

::set batch_size=20 如果你爆显存了 请将batch_size调低
set batch_size=15
set grad_accumulate=2
set lr=5e-4
set l2=1e-4
set smooth=0.15
set warmup_ratio=0.1
set lr_schedule=linear
set eval_after_epoch=60
set max_epoch=100
set max_norm=5
set beam_size=5

python scripts/text2sql.py --task %task% --seed %seed% --device %device% %testing% %read_model_path% ^
    --gnn_hidden_size %gnn_hidden_size% --dropout %dropout% --attn_drop %attn_drop% --att_vec_size %att_vec_size% ^
    --model %model% --output_model %output_model% --local_and_nonlocal %local_and_nonlocal% --score_function %score_function% %relation_share_heads% ^
    --schema_aggregation %schema_aggregation% --embed_size %embed_size% --gnn_num_layers %gnn_num_layers% --num_heads %num_heads% %sep_cxt% ^
    --lstm %lstm% --chunk_size %chunk_size% --drop_connect %drop_connect% --lstm_hidden_size %lstm_hidden_size% --lstm_num_layers %lstm_num_layers% ^
    --action_embed_size %action_embed_size% --field_embed_size %field_embed_size% --type_embed_size %type_embed_size% ^
    %no_context_feeding% %no_parent_production_embed% %no_parent_field_embed% %no_parent_field_type_embed% %no_parent_state% ^
    --batch_size %batch_size% --grad_accumulate %grad_accumulate% --lr %lr% --l2 %l2% --warmup_ratio %warmup_ratio% --lr_schedule %lr_schedule% --eval_after_epoch %eval_after_epoch% ^
    --smooth %smooth% --max_epoch %max_epoch% --max_norm %max_norm% --beam_size %beam_size%
