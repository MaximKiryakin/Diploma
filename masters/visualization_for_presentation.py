# ВИЗУАЛИЗАЦИЯ ДЛЯ ПРЕЗЕНТАЦИИ НА СЕМИНАРЕ
# Создание графиков, демонстрирующих управление кредитным портфелем

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_risk_management_demo():
    """
    Создает демонстрационный график: портфель без управления vs с управлением
    """
    
    # Симулируем данные: PD портфеля за 2 года
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='M')
    
    # Базовая PD (без управления)
    base_pd = 0.03 + 0.02 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.005, len(dates))
    base_pd = np.clip(base_pd, 0.01, 0.08)
    
    # PD с управлением (более стабильная, ниже пики)
    managed_pd = base_pd * 0.7 + 0.005  # Снижение на 30% + стабилизация
    managed_pd = np.clip(managed_pd, 0.01, 0.05)  # Жесткие лимиты
    
    # Макроэкономические шоки (для контекста)
    shocks = np.zeros(len(dates))
    shocks[8] = 0.015   # Кризис весна 2022
    shocks[15] = 0.012  # Волатильность зима 2022-23
    shocks[20] = 0.008  # Летний кризис 2023
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # График 1: Сравнение портфелей
    ax1.plot(dates, base_pd * 100, 'o-', linewidth=3, markersize=6, 
             color='red', alpha=0.8, label='Портфель БЕЗ управления')
    ax1.plot(dates, managed_pd * 100, 's-', linewidth=3, markersize=6,
             color='green', alpha=0.8, label='Портфель С управлением')
    
    # Добавляем шоки
    for i, shock in enumerate(shocks):
        if shock > 0:
            ax1.axvline(dates[i], color='orange', alpha=0.6, linestyle='--', linewidth=2)
            ax1.text(dates[i], max(base_pd)*100 + 0.5, 'Макро\nшок', 
                    ha='center', va='bottom', fontsize=10, color='orange', fontweight='bold')
    
    ax1.axhline(5, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Лимит PD (5%)')
    ax1.fill_between(dates, 0, 5, alpha=0.1, color='green', label='Безопасная зона')
    ax1.fill_between(dates, 5, max(base_pd)*100 + 1, alpha=0.1, color='red', label='Зона риска')
    
    ax1.set_title('ЭФФЕКТИВНОСТЬ УПРАВЛЕНИЯ КРЕДИТНЫМ ПОРТФЕЛЕМ', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Портфельная PD (%)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(base_pd)*100 + 1)
    
    # График 2: Экономический эффект
    # Расчет потерь (PD * средний размер кредита * количество кредитов)
    portfolio_size = 1000  # млн руб
    losses_unmanaged = base_pd * portfolio_size
    losses_managed = managed_pd * portfolio_size
    
    ax2.fill_between(dates, losses_unmanaged, color='red', alpha=0.3, label='Потери без управления')
    ax2.fill_between(dates, losses_managed, color='green', alpha=0.3, label='Потери с управлением')
    ax2.plot(dates, losses_unmanaged, color='red', linewidth=2)
    ax2.plot(dates, losses_managed, color='green', linewidth=2)
    
    # Экономия
    savings = losses_unmanaged - losses_managed
    total_savings = np.sum(savings)
    
    ax2.fill_between(dates, losses_managed, losses_unmanaged, 
                    color='blue', alpha=0.2, label=f'Экономия: {total_savings:.1f} млн руб.')
    
    ax2.set_title('ЭКОНОМИЧЕСКИЙ ЭФФЕКТ ОТ УПРАВЛЕНИЯ РИСКАМИ', 
                 fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Период', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ожидаемые потери (млн руб.)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('presentation/risk_management_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return total_savings

def create_decision_process_flow():
    """
    Создает схему процесса принятия кредитных решений
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Этапы процесса
    stages = [
        {'name': 'ЗАЯВКА\nНА КРЕДИТ', 'pos': (2, 8), 'color': 'lightblue'},
        {'name': 'РАСЧЕТ PD\n(Модель Мертона)', 'pos': (6, 8), 'color': 'lightgreen'},
        {'name': 'ПРОВЕРКА\nЛИМИТОВ', 'pos': (10, 8), 'color': 'lightyellow'},
        {'name': 'ОЦЕНКА\nКОНЦЕНТРАЦИИ', 'pos': (14, 8), 'color': 'lightcoral'},
        
        {'name': 'PD ≤ 5%?', 'pos': (6, 5), 'color': 'yellow'},
        {'name': 'Доля ≤ 8%?', 'pos': (10, 5), 'color': 'yellow'},
        {'name': 'Сектор ≤ 25%?', 'pos': (14, 5), 'color': 'yellow'},
        
        {'name': '✅ ОДОБРИТЬ', 'pos': (4, 2), 'color': 'lightgreen'},
        {'name': '⚠️ УСЛОВНО', 'pos': (8, 2), 'color': 'orange'},
        {'name': '❌ ОТКЛОНИТЬ', 'pos': (12, 2), 'color': 'lightcoral'},
    ]
    
    # Рисуем блоки
    for stage in stages:
        x, y = stage['pos']
        rect = plt.Rectangle((x-1, y-0.5), 2, 1, 
                           facecolor=stage['color'], 
                           edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, stage['name'], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Стрелки
    arrows = [
        ((3, 8), (5, 8)),    # Заявка → PD
        ((7, 8), (9, 8)),    # PD → Лимиты
        ((11, 8), (13, 8)),  # Лимиты → Концентрация
        
        ((6, 7.5), (6, 5.5)), # PD → Проверка PD
        ((10, 7.5), (10, 5.5)), # Лимиты → Проверка доли
        ((14, 7.5), (14, 5.5)), # Концентрация → Проверка сектора
        
        ((5, 4.5), (4, 2.5)),   # Да → Одобрить
        ((7, 4.5), (8, 2.5)),   # Частично → Условно
        ((11, 4.5), (12, 2.5)), # Нет → Отклонить
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Добавляем лимиты
    limits_text = """
    ЛИМИТЫ РИСКА:
    • Максимальная PD: 5%
    • Доля заемщика: 8%
    • Секторная концентрация: 25%
    
    СТАВКА:
    AAA (PD≤1%): 8%
    AA (PD≤2%): 10%
    A (PD≤3%): 12%
    BBB (PD≤5%): 15%
    """
    
    ax.text(16, 6, limits_text, fontsize=11, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_title('СИСТЕМА ПРИНЯТИЯ КРЕДИТНЫХ РЕШЕНИЙ', 
                fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('presentation/decision_process.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_portfolio_optimization():
    """
    Создает график оптимизации портфеля
    """
    # Данные о компаниях
    companies = ['SBER', 'GAZP', 'VTBR', 'LKOH', 'GMKN', 'ROSN', 'NLMK', 'MTSS']
    pd_values = [0.015, 0.025, 0.045, 0.028, 0.035, 0.032, 0.055, 0.038]
    returns = [0.09, 0.10, 0.12, 0.105, 0.11, 0.108, 0.13, 0.14]
    amounts = [100, 150, 80, 200, 120, 90, 70, 60]  # млн руб
    
    # Цветовая карта по рискам
    colors = ['green' if pd < 0.03 else 'yellow' if pd < 0.05 else 'red' for pd in pd_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # График 1: Карта риск-доходность
    scatter = ax1.scatter(np.array(pd_values)*100, np.array(returns)*100, 
                         s=[a*3 for a in amounts], c=colors, alpha=0.7, edgecolors='black')
    
    for i, comp in enumerate(companies):
        ax1.annotate(comp, (pd_values[i]*100, returns[i]*100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Добавляем границы
    ax1.axvline(5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Лимит PD (5%)')
    ax1.axvline(3, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Целевой PD (3%)')
    
    ax1.set_xlabel('Вероятность дефолта (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ожидаемая доходность (%)', fontsize=12, fontweight='bold')
    ax1.set_title('КАРТА РИСК-ДОХОДНОСТЬ ЗАЕМЩИКОВ', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Оптимизированный портфель
    # Выбираем компании с PD < 5%
    selected_mask = np.array(pd_values) < 0.05
    selected_companies = [companies[i] for i in range(len(companies)) if selected_mask[i]]
    selected_amounts = [amounts[i] for i in range(len(amounts)) if selected_mask[i]]
    
    # Нормализуем размеры для бюджета 1000 млн
    total_requested = sum(selected_amounts)
    budget = 800  # млн руб
    allocated_amounts = [a * budget / total_requested for a in selected_amounts]
    
    wedges, texts, autotexts = ax2.pie(allocated_amounts, labels=selected_companies, autopct='%1.1f%%',
                                      colors=plt.cm.Set3(np.linspace(0, 1, len(selected_companies))))
    
    ax2.set_title('ОПТИМИЗИРОВАННЫЙ ПОРТФЕЛЬ\n(Бюджет: 800 млн руб.)', 
                 fontsize=14, fontweight='bold')
    
    # Добавляем метрики
    selected_pd = [pd_values[i] for i in range(len(pd_values)) if selected_mask[i]]
    selected_returns = [returns[i] for i in range(len(returns)) if selected_mask[i]]
    
    portfolio_pd = np.average(selected_pd, weights=allocated_amounts)
    portfolio_return = np.average(selected_returns, weights=allocated_amounts)
    
    metrics_text = f"""
    ПОРТФЕЛЬНЫЕ МЕТРИКИ:
    • Средняя PD: {portfolio_pd:.3f} ({portfolio_pd*100:.1f}%)
    • Ожидаемая доходность: {portfolio_return:.2%}
    • Риск-премия: {portfolio_return-portfolio_pd:.2%}
    • Коэффициент Шарпа: {(portfolio_return-0.08)/0.02:.2f}
    • Использование бюджета: 80%
    """
    
    ax2.text(1.3, 0, metrics_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('presentation/portfolio_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return portfolio_pd, portfolio_return

def create_macro_impact_demo():
    """
    Демонстрация влияния макрошоков на PD
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Данные для демонстрации
    periods = np.arange(0, 20, 1)  # 20 кварталов
    
    # График 1: Шок инфляции
    baseline_pd = np.full(20, 0.03)  # Базовый уровень 3%
    inflation_shock = np.zeros(20)
    inflation_shock[5:] = 0.015 * np.exp(-0.3 * np.arange(15))  # Шок на 5-м периоде
    
    ax1.plot(periods, baseline_pd * 100, '--', color='blue', linewidth=2, label='Базовый уровень')
    ax1.plot(periods, (baseline_pd + inflation_shock) * 100, '-o', color='red', linewidth=3, 
             markersize=6, label='После шока инфляции')
    ax1.axvline(5, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    ax1.text(5.5, 4, 'Шок\n+2 п.п.', fontsize=10, color='orange', fontweight='bold')
    ax1.set_title('Влияние шока инфляции на PD', fontweight='bold')
    ax1.set_ylabel('PD (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Шок ключевой ставки
    rate_shock = np.zeros(20)
    rate_shock[8:] = 0.012 * (1 - np.exp(-0.4 * np.arange(12)))  # Постепенное нарастание
    
    ax2.plot(periods, baseline_pd * 100, '--', color='blue', linewidth=2, label='Базовый уровень')
    ax2.plot(periods, (baseline_pd + rate_shock) * 100, '-s', color='green', linewidth=3,
             markersize=6, label='После роста ставки')
    ax2.axvline(8, color='purple', linestyle=':', alpha=0.7, linewidth=2)
    ax2.text(8.5, 4, 'Рост ставки\n+3 п.п.', fontsize=10, color='purple', fontweight='bold')
    ax2.set_title('Влияние роста ключевой ставки', fontweight='bold')
    ax2.set_ylabel('PD (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Шок валютного курса (сектора по-разному)
    # Экспортеры (нефтегаз) - снижение PD
    # Импортеры (ритейл) - рост PD
    usd_shock_export = np.zeros(20)
    usd_shock_import = np.zeros(20)
    usd_shock_export[10:] = -0.008 * np.exp(-0.2 * np.arange(10))  # Снижение для экспортеров
    usd_shock_import[10:] = 0.020 * (1 - np.exp(-0.5 * np.arange(10)))  # Рост для импортеров
    
    ax3.plot(periods, baseline_pd * 100, '--', color='blue', linewidth=2, label='Базовый уровень')
    ax3.plot(periods, (baseline_pd + usd_shock_export) * 100, '-^', color='green', linewidth=3,
             markersize=6, label='Экспортеры (ГАЗП)')
    ax3.plot(periods, (baseline_pd + usd_shock_import) * 100, '-v', color='red', linewidth=3,
             markersize=6, label='Импортеры (MGNT)')
    ax3.axvline(10, color='gold', linestyle=':', alpha=0.7, linewidth=2)
    ax3.text(10.5, 4.5, 'Девальвация\n+20 руб/$', fontsize=10, color='gold', fontweight='bold')
    ax3.set_title('Влияние девальвации рубля', fontweight='bold')
    ax3.set_xlabel('Период (кварталы)')
    ax3.set_ylabel('PD (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # График 4: Комбинированный эффект
    combined_shock = inflation_shock + rate_shock + usd_shock_import
    
    ax4.fill_between(periods, baseline_pd * 100, (baseline_pd + combined_shock) * 100, 
                    alpha=0.3, color='red', label='Зона повышенного риска')
    ax4.plot(periods, baseline_pd * 100, '--', color='blue', linewidth=2, label='Базовый уровень')
    ax4.plot(periods, (baseline_pd + combined_shock) * 100, '-o', color='red', linewidth=3,
             markersize=6, label='Комбинированный эффект')
    ax4.axhline(5, color='darkred', linestyle='-', alpha=0.8, linewidth=2, label='Критический уровень (5%)')
    
    ax4.set_title('Системный риск: комбинированный эффект шоков', fontweight='bold')
    ax4.set_xlabel('Период (кварталы)')
    ax4.set_ylabel('PD (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('ИМПУЛЬСНЫЕ ОТКЛИКИ: ВЛИЯНИЕ МАКРОЭКОНОМИЧЕСКИХ ШОКОВ НА PD', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('presentation/macro_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Создание визуализаций для презентации...")
    
    # Создаем директорию
    import os
    os.makedirs('presentation', exist_ok=True)
    
    # Генерируем графики
    print("\n1. График сравнения управляемого и неуправляемого портфеля...")
    savings = create_risk_management_demo()
    
    print(f"\n2. Схема процесса принятия решений...")
    create_decision_process_flow()
    
    print(f"\n3. Оптимизация портфеля...")
    portfolio_pd, portfolio_return = create_portfolio_optimization()
    
    print(f"\n4. Анализ макроэкономических шоков...")
    create_macro_impact_demo()
    
    print(f"\n✅ Все графики созданы в папке 'presentation/'")
    print(f"\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ ДЛЯ ПРЕЗЕНТАЦИИ:")
    print(f"   • Экономия от управления: {savings:.1f} млн руб. за 2 года")
    print(f"   • Портфельная PD: {portfolio_pd*100:.1f}% (vs лимит 5%)")  
    print(f"   • Портфельная доходность: {portfolio_return:.1%}")
    print(f"   • Снижение рисков: до 30% при макрошоках")