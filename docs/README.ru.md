# Grokking модульной арифметики: MLP и spline-KAN

Этот проект проверяет не вопрос «какая сеть точнее», а более узкий вопрос:
как меняется порядок фаз запоминания и обобщения, если в архитектурной основе
по Громову заменить quadratic MLP на spline-KAN сопоставимого размера. Это
адаптированное сравнение на AdamW, а не точное воспроизведение оптимизации из
работы Громова.

## Главный результат

После exploratory-поиска на `p=97` были зафиксированы критерий и подтверждающая
матрица из 80 запусков: четыре условия, две модели, два значения weight decay и
пять seed.

- Во всей матрице MLP показала классический grokking в 30/40 запусков, KAN --
  в 8/40.
- Главный локальный контраст: при `p=31`, 35% train MLP дала 0/10, а KAN --
  8/10, по 4/5 для каждого weight decay. При 50% данных обобщение стало совместным или
  слабозадержанным; все 20 запусков `p=53` классифицированы как weak delay.
- На symmetry-safe split KAN в этом успешном random-split режиме дала 0/5,
  тогда как MLP при `p=31`, 50% сохранила classical grokking в 5/5.
- Поэтому вывод проекта не «KAN лучше MLP». KAN сдвигает границы между
  запоминанием, плато и обобщением, причём результат зависит от данных,
  регуляризации и split.

Строгий критерий, заранее зафиксированная матрица и история изменения гипотезы
описаны в [research_protocol.md](docs/research_protocol.md).

## Быстрый запуск

Нужен Python 3.11 или 3.12. Финальная проверка выполнена на Python 3.12.13.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_experiment.py `
  --config configs/gromov_quadratic_adamw.yaml `
  --output-dir results/core_runs/smoke `
  --set training.max_steps=200 `
  --set training.eval_every=20
python scripts/make_figures.py --run-dir results/core_runs/smoke
```

Каждый override передаётся отдельным `--set`. Smoke-run проверяет полный путь
данные -> обучение -> артефакты -> график; он не предназначен для получения
grokking за 200 шагов.

## Полное воспроизведение

```powershell
python scripts/run_study.py `
  --manifest configs/final_study.yaml `
  --output-dir results/core_runs `
  --workers 4
python scripts/aggregate_study.py --input-dir results/core_runs
python scripts/analyze_mechanism_matrix.py `
  --study-dir results/core_runs
python scripts/audit_study.py
```

`run_study.py` не перезаписывает завершённые запуски. Manifest содержит полную
конфигурацию, seed, относительный путь, статус и время. В основной серии нет
досрочной остановки: `p=31` обучается 15 000 шагов, `p=53` -- 20 000.
`audit_study.py` проверяет всю зафиксированную матрицу, фазовую статистику,
цензурирование, механизм, ablation-повторы и контрольные суммы, а не только
наличие файлов.

## Где лежат результаты

- `results/tables/paired_runs.csv` -- все 80 индивидуальных результатов;
- `results/tables/paired_summary.csv` -- 16 ячеек, Wilson intervals и
  цензурирование;
- `results/figures/paired_phase_heatmap.png` -- главная фазовая карта;
- `results/figures/paired_delays.png` -- все seed и медианы задержек;
- `results/tables/mechanism_*.csv` -- Fourier и PCA для 10 моделей;
- `results/tables/fourier_spectra.csv` -- спектр по частотам `(f_a, f_b)` для
  каждого seed и checkpoint;
- `results/tables/fourier_top_modes.csv` -- двенадцать ведущих мод каждого
  спектра;
- `results/figures/fourier_spectrum_heatmaps.png` -- медианная волновая
  структура по моделям и фазам;
- `results/figures/fourier_runs/` -- индивидуальные heatmap десяти моделей;
- `results/tables/edge_ablation_*.csv` -- top/random/bottom ablation пяти KAN;
- `results/tables/symmetry_phase_summary.csv` и
  `results/figures/symmetry_phase_comparison.png` -- явное сопоставление
  random и symmetry-safe результатов;
- `results/core_runs/` -- config, metadata, history и summary каждого запуска.

Веса и snapshots сохранены только для десяти заранее выбранных
механистических запусков `p=31`, 35%, `wd=0.3`. Карта исходников находится в
[code_map.md](docs/code_map.md).

## Что считается grokking

99% test accuracy недостаточно. Классический grokking требует устойчивого
train-запоминания, test не выше 20% в этой точке, разрыва не меньше 50
процентных пунктов, плато не менее 1000 шагов и только затем устойчивых 99%
test. Остальные запуски получают отдельные метки; незавершившееся обобщение
остаётся цензурированным наблюдением.

## Ограничения

- сравниваются две конкретные ширины и один B-spline basis;
- пять seed дают широкие Wilson intervals;
- MLP воспроизводит архитектурную основу Громова, но основная матрица использует
  AdamW и не является точной репликацией его gradient-descent протокола;
- `p=97` не используется как главное доказательство;
- результат KAN чувствителен к симметрии train/test split;
- Fourier-концентрация и edge-ablation не доказывают, что сеть «поняла
  арифметику» или использует единственный внутренний алгоритм.

Научный отчёт, краткое описание и редактируемая презентация находятся в
`docs/`.
