#!/usr/bin/env python
"""
Пример использования функций активного управления портфелем
Portfolio Management Example
"""

from utils.portfolio import Portfolio


def main():
    """
    Демонстрация возможностей активного управления портфелем
    """

    print("Portfolio Management System - Demo")
    print("=" * 50)

    # Создание экземпляра портфеля
    tickers_list = [
        "GAZP",
        "LKOH",
        "ROSN",
        "SBER",
        "VTBR",
        "MOEX",
        "GMKN",
        "NLMK",
        "RUAL",
        "MTSS",
        "RTKM",
        "TTLK",
    ]

    calc = Portfolio(
        dt_calc="2025-05-31",
        dt_start="2019-11-03",
        stocks_step=10,
        tickers_list=tickers_list,
    )

    try:
        # Базовый анализ портфеля
        print("\n1. Загрузка и подготовка данных...")
        calc = (
            calc.log_system_info()
            .load_stock_data(use_backup_data=True, create_backup=False)
            .load_multipliers()
            .create_portfolio()
            .adjust_portfolio_data_types()
            .add_macro_data()
            .fill_missing_values()
            .add_dynamic_features()
            .add_merton_pd()
        )

        print("✓ Данные загружены и обработаны")

        # Активное управление портфелем
        print("\n2. Активное управление портфелем...")

        # Расчет доходностей
        calc.calculate_portfolio_returns()
        print("✓ Доходности рассчитаны")

        # Оптимизация по минимальной дисперсии
        calc.optimize_portfolio_weights(method="min_variance")
        print("✓ Оптимизация (минимальная дисперсия) выполнена")

        # Альтернативная оптимизация по Шарпу
        calc_sharpe = Portfolio(
            dt_calc="2025-05-31",
            dt_start="2019-11-03",
            stocks_step=10,
            tickers_list=tickers_list,
        )

        # Копируем уже подготовленные данные
        calc_sharpe.portfolio = calc.portfolio.copy()
        calc_sharpe.calculate_portfolio_returns()
        calc_sharpe.optimize_portfolio_weights(method="max_sharpe", risk_free_rate=0.08)
        print("✓ Оптимизация (максимальный Шарп) выполнена")

        # Визуализация результатов
        calc.plot_optimal_weights(verbose=False)
        print("✓ Графики сохранены")

        # Расчет портфельного PD
        calc.calculate_risk_adjusted_pd()
        print("✓ Портфельный PD рассчитан")

        # Симуляция ребалансировки
        print("\n3. Анализ ребалансировки...")
        current_weights = {ticker: 1 / len(tickers_list) for ticker in tickers_list}
        calc.rebalance_portfolio(
            current_weights=current_weights, rebalancing_threshold=0.02
        )

        # Результаты
        print("\n4. Результаты оптимизации:")
        print("-" * 30)

        summary_min_var = calc.get_portfolio_summary()
        summary_max_sharpe = calc_sharpe.get_portfolio_summary()

        print(f"Минимальная дисперсия:")
        expected_return = summary_min_var.get('expected_return', 'N/A')
        print(f"  Ожидаемая доходность: {expected_return:.4f}")
        volatility = summary_min_var.get('volatility', 'N/A')
        print(f"  Волатильность: {volatility:.4f}")
        sharpe_ratio = summary_min_var.get('sharpe_ratio', 'N/A')
        print(f"  Коэффициент Шарпа: {sharpe_ratio:.4f}")

        print(f"\nМаксимальный Шарп:")
        expected_return_max = summary_max_sharpe.get('expected_return', 'N/A')
        print(f"  Ожидаемая доходность: {expected_return_max:.4f}")
        volatility_max = summary_max_sharpe.get('volatility', 'N/A')
        print(f"  Волатильность: {volatility_max:.4f}")
        sharpe_ratio_max = summary_max_sharpe.get('sharpe_ratio', 'N/A')
        print(f"  Коэффициент Шарпа: {sharpe_ratio_max:.4f}")

        # Топ активы по весам
        if calc.optimal_weights is not None:
            print(f"\nТоп-5 активов (минимальная дисперсия):")
            top_weights = calc.optimal_weights.nlargest(5)
            for ticker, weight in top_weights.items():
                print(f"  {ticker}: {weight:.3f}")

        if calc_sharpe.optimal_weights is not None:
            print(f"\nТоп-5 активов (максимальный Шарп):")
            top_weights_sharpe = calc_sharpe.optimal_weights.nlargest(5)
            for ticker, weight in top_weights_sharpe.items():
                print(f"  {ticker}: {weight:.3f}")

        # Информация о ребалансировке
        if hasattr(calc, "rebalancing_trades") and calc.rebalancing_trades:
            print(f"\n5. Рекомендации по ребалансировке:")
            print("-" * 35)
            for ticker, trade_info in calc.rebalancing_trades.items():
                action = trade_info["action"]
                size = abs(trade_info["trade_size"])
                current_w = trade_info['current_weight']
                target_w = trade_info['target_weight']
                print(f"  {ticker}: {action} {size:.3f} ({current_w:.3f} → {target_w:.3f})")
        else:
            print("\n5. Ребалансировка не требуется")

        calc.log_completion()
        print("\n✓ Анализ завершен успешно!")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
