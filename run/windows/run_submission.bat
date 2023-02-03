set saved_model=saved_models\%1
set output_path=saved_models\%1\predicted_sql.txt
set batch_size=10
set beam_size=5
set current_dir=%cd%

python eval.py --db_dir %current_dir%\data\database --table_path %current_dir%\data\tables.json --dataset_path %current_dir%\data\dev.json --saved_model %saved_model% --output_path %output_path% --batch_size %batch_size% --beam_size %beam_size%
python evaluation.py --gold %current_dir%\data\dev_gold.sql --pred %output_path% --db %current_dir%\data\database --table %current_dir%\data\tables.json --etype match > %saved_model%\evaluation.log
