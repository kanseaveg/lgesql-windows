
set current_dir=%cd%

set train_data=%current_dir%\data\train.json
set dev_data=%current_dir%\data\dev.json
set table_data=%current_dir%\data\tables.json
set train_out=%current_dir%\data\train.lgesql.bin
set dev_out=%current_dir%\data\dev.lgesql.bin
set table_out=%current_dir%\data\tables.bin
set vocab_glove=%current_dir%\pretrained_models\glove.42b.300d\vocab_glove.txt
set vocab=%current_dir%\pretrained_models\glove.42b.300d\vocab.txt

echo "Start to preprocess the original train dataset ..."
python -u preprocess/process_dataset.py --dataset_path %train_data% --raw_table_path %table_data% --table_path %table_out% --output_path %current_dir%\data\train.bin --skip_large
echo "Start to preprocess the original dev dataset ..."
python -u preprocess/process_dataset.py --dataset_path %dev_data% --table_path %table_out% --output_path %current_dir%\data\dev.bin
echo "Start to build word vocab for the dataset ..."
python -u preprocess/build_glove_vocab.py --data_paths %current_dir%\data\train.bin --table_path %table_out% --reference_file %vocab_glove% --mwf 4 --output_path %vocab%
echo "Start to construct graphs for the dataset ..."
python -u preprocess/process_graphs.py --dataset_path %current_dir%\data\train.bin --table_path %table_out% --method lgesql --output_path %train_out%
python -u preprocess/process_graphs.py --dataset_path %current_dir%\data\dev.bin --table_path %table_out% --method lgesql --output_path %dev_out%
