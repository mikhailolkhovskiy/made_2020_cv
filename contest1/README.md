01_train_full_resnext101_32x8d - бейзлайн с использованием resnext101_32x8d

02_tain_last_fc - добавил ReduceLROnPlateau и заморозил все слои модели кроме последнего 

Как тренеровал модель:
1) Запустил скрипт из 01_train_full_resnext101_32x8d на 6 эпох 
2) Продолжил тренеровку модели (1) скриптом 02_tain_last_fc 10 эпох
3) Продолжил тренеровку модели (2) скриптом 02_tain_last_fc 6 эпох

![Submissions](submissions.png)
