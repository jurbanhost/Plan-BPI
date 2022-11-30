
<h1 align="center">Решение II места для соревнования <a href="https://hacks-ai.ru/championships/758301">Чемпионат, кейс План БТИ</a> 

## Для воспроизведения результатов выполните следующие действия:

#### Запускаем обучение для каждой модели
```
python main.py --cfg configs/baseline.yml
```

#### Генерируем предикты

```
python generate_predictions.py 
python generate_predictions.py 
python generate_predictions.py 

python generate_predictions.py --cfg experiments/segformer/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>


python generate_predictions.py --cfg experiments/segformer/baseline.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>
```

####
#### Запускаем скрипт, который усреднит все предикты и сгенерирует итоговую маску
```
python create_submission_ensemble.py
```
