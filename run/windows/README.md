### Preprocess Data

```cmd
:: please prepare the spider train.json file
.\run\windows\run_preprocessing.bat
```

### Train
```cmd
:: mmc
.\run\windows\run_lgesql_glove.bat mmc

:: msde
.\run\windows\run_lgesql_glove.bat msde
```