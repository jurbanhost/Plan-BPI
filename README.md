
<h1 align="center">Решение N места для соревнования <a href="https://hacks-ai.ru/championships/758301">Чемпионат, кейс План БТИ</a> 

## Для воспроизведения результатов выполните следующие действия:

#### Запускаем обучение для каждой модели
```
python main.py --cfg configs/segformer_1024_b4_g16_adamW_cosine.yml
python main.py --cfg configs/segformer_1024_b4_adamW_cosine.yml
python main.py --cfg configs/segformer_864_b4_8_cosine_light.yml
python main.py --cfg configs/segformer_864_b4_8_cosine_dark.yml
```

#### Генерируем предикты

```
python generate_predictions.py 
python generate_predictions.py 
python generate_predictions.py 

python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.651



python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>
python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.635
```

####
#### Запускаем скрипт, который усреднит все предикты и сгенерирует итоговую маску
```
python create_submission_ensemble.py
```
