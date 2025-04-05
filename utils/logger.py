import logging
import sys
import os
from datetime import datetime
from typing import Optional

class Logger:
    """
    Класс-обертка над стандартным logging с записью в общий файл.
    """

    # Общий файл логов для всех экземпляров класса
    _log_file = f"logs/app_{datetime.now()}.log"
    _initialized = False

    def __init__(
            self,
            name: str,
            level: int = logging.INFO,
            format: str = "%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    ):
        """
        Инициализация логгера.

        :param name: имя логгера (обычно __name__)
        :param level: уровень логирования (по умолчанию INFO)
        :param format: формат сообщений
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Проверяем, был ли уже настроен обработчик файла
        if not Logger._initialized:
            self._setup_handlers(format)
            Logger._initialized = True

    def _setup_handlers(self, format: str):
        """Настройка обработчиков логов."""
        formatter = logging.Formatter(format)

        # Создаем директорию для логов, если ее нет
        os.makedirs(os.path.dirname(Logger._log_file), exist_ok=True)

        # Обработчик для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Обработчик для записи в файл
        file_handler = logging.FileHandler(Logger._log_file)
        file_handler.setFormatter(formatter)

        # Добавляем обработчики к корневому логгеру
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Возвращает настроенный логгер"""
        return self.logger