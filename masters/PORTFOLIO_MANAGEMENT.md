# Руководство по активному управлению портфелем

## Новые возможности

Добавлены следующие методы управления портфелем в класс `Portfolio`:

### 1. Расчет доходностей активов
```python
calc.calculate_portfolio_returns()
```
Вычисляет матрицу доходностей активов на основе исторических цен.

### 2. Оптимизация весов портфеля
```python
# Минимизация риска (дисперсии)
calc.optimize_portfolio_weights(method='min_variance')

# Максимизация коэффициента Шарпа
calc.optimize_portfolio_weights(method='max_sharpe', risk_free_rate=0.08)

# Risk Parity (равный риск-вклад)
calc.optimize_portfolio_weights(method='risk_parity')
```

### 3. Ребалансировка портфеля
```python
current_weights = {'GAZP': 0.15, 'SBER': 0.20, ...}
calc.rebalance_portfolio(
    current_weights=current_weights,
    rebalancing_threshold=0.05  # Порог в 5%
)
```

### 4. Портфельный анализ рисков
```python
# Расчет портфельного PD с учетом диверсификации
calc.calculate_risk_adjusted_pd(portfolio_weights=weights)
```

### 5. Визуализация результатов
```python
# График оптимальных весов (столбчатая диаграмма + круговая)
calc.plot_optimal_weights(verbose=False)
```

### 6. Сводная информация
```python
summary = calc.get_portfolio_summary()
print(f"Ожидаемая доходность: {summary['expected_return']:.4f}")
print(f"Волатильность: {summary['volatility']:.4f}")
print(f"Коэффициент Шарпа: {summary['sharpe_ratio']:.4f}")
```

## Пример полного рабочего процесса

```python
from utils.portfolio import Portfolio

# 1. Создание и настройка портфеля
calc = Portfolio(
    dt_calc='2025-05-31',
    dt_start='2019-11-03',
    stocks_step=10,
    tickers_list=['GAZP', 'SBER', 'LKOH', ...]
)

# 2. Базовая подготовка данных
calc = (
    calc
    .load_stock_data(use_backup_data=True)
    .load_multipliers()
    .create_portfolio()
    .adjust_portfolio_data_types()
    .add_macro_data()
    .add_merton_pd()
)

# 3. Активное управление
calc = (
    calc
    .calculate_portfolio_returns()              # Расчет доходностей
    .optimize_portfolio_weights(                # Оптимизация весов
        method='max_sharpe', 
        risk_free_rate=0.08
    )
    .plot_optimal_weights(verbose=False)        # Визуализация
    .calculate_risk_adjusted_pd()               # Портфельный PD
)

# 4. Ребалансировка
current_portfolio = {...}  # Текущие веса
calc.rebalance_portfolio(
    current_weights=current_portfolio,
    rebalancing_threshold=0.03
)

# 5. Получение результатов
summary = calc.get_portfolio_summary()
```

## Интерпретация результатов

### Методы оптимизации:
- **min_variance**: Минимизирует риск портфеля, подходит для консервативных инвесторов
- **max_sharpe**: Максимизирует отношение доходность/риск, сбалансированный подход  
- **risk_parity**: Равный вклад в риск от каждого актива, хорошая диверсификация

### Ключевые метрики:
- **Expected Return**: Ожидаемая годовая доходность портфеля
- **Volatility**: Годовая волатильность (стандартное отклонение доходности)
- **Sharpe Ratio**: Коэффициент Шарпа (доходность сверх безрискового актива на единицу риска)
- **Portfolio PD**: Средневзвешенная вероятность дефолта портфеля

### Сигналы для ребалансировки:
Если отклонение текущих весов от целевых превышает пороговое значение (например, 5%), 
система рекомендует торговые операции для восстановления оптимального распределения.

## Файлы результатов

Все графики сохраняются в папку `logs/graphs/`:
- `optimal_weights.png` - График оптимальных весов портфеля
- Остальные графики из базового анализа (PD, корреляции, etc.)

## Пример запуска

Для быстрого тестирования запустите:
```bash
python portfolio_management_example.py
```

Этот скрипт продемонстрирует все новые возможности управления портфелем.