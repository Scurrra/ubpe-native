import sys
import time


class Progress:
    """Progress bar for tracking progress of a task."""

    _logger: "Logger | None"
    _unit: str
    _precision: int
    _is_running: bool = False
    _is_active: bool = False
    _total: int | None = None
    _initial: int | None = None
    _current: int | None = None
    _initial_time: float | None = None
    _current_time: float | None = None
    _rate: float | None = None
    _old_length: int | None = None

    def __init__(
        self,
        *,
        logger: "Logger | None" = None,
        unit: str | None = None,
        precision: int | None = None,
    ):
        """Initialize the progress meter instance.

        Args:
            logger (Logger | None): Logger to use for logging progress. If `None`, the process will be logged to stdout.
            unit (str | None): Unit of measurement for progress. Default is "item".
            precision (int | None): Precision of average progress speed to display. Default is 3.
        """

        if unit is None or not isinstance(unit, str):
            unit = "item"
        if precision is None or not isinstance(precision, int):
            precision = 3
        self._unit = unit
        self._precision = precision
        self._logger = logger
        self._is_active = False
        self._is_running = False

    def __call__(self, *, total: int, initial: int = 0):
        """Initialize the progress meter.

        Args:
            total (int): Total number of items to process.
            initial (int): Initial number of items processed.

        Note: If used in a for loop, the progress meter will be automatically started (no call `.run()` needed).
        """

        if self._is_running:
            raise Exception("Progress is already running")

        self._total = total
        self._initial = initial
        self._current = initial
        self._initial_time = time.time()
        self._current_time = self._initial_time
        self._rate = 0.0
        self._old_length = None
        self._is_active = True
        return self

    def _reset(self):
        if self._is_running:
            raise Exception("Progress is already running")

        self._total = None
        self._initial = None
        self._current = None
        self._initial_time = None
        self._current_time = None
        self._rate = None
        self._old_length = None
        self._is_active = False

    def run(self):
        """Start the progress meter."""

        if self._is_running:
            raise Exception("Progress is already running")

        if not self._is_active:
            raise Exception("Progress is not active")

        self._is_running = True

        if self._logger is not None:
            self._old_length = self._logger.log_progress()
        else:
            print(
                f"{self._current} / {self._total} [{self._rate:.{self._precision}f} {self._unit}s/sec]"
            )

    def stop(self):
        """Stop the progress meter."""

        self._is_running = False
        self._reset()

    def __iter__(self):
        if self._is_running:
            raise Exception("Progress is already running")

        if not self._is_active:
            raise Exception("Progress is not active")

        self._is_running = True
        return self

    def __next__(self):
        if self._current > self._total:  # type: ignore (no `None` here)
            self.stop()
            raise StopIteration
        item = self._current
        self._current_time = time.time()

        elapsed = self._current_time - self._initial_time  # type: ignore (no `None` here)
        self._rate = (self._current - self._initial) / elapsed  # type: ignore (no `None` here)

        if self._logger is not None:
            self._old_length = self._logger.log_progress()
        else:
            left = (
                (self._total - self._current) / self._rate if self._rate > 0.0 else 0.0  # type: ignore (no `None` here)
            )
            estimated = elapsed if left < 0.0 else elapsed + left
            time_str = f"{int(elapsed // 60)}:{elapsed % 60:.0f}<{int(estimated // 60)}:{estimated % 60:.0f}"
            if self._rate >= 1.0 or self._rate == 0.0:
                print(
                    f"{self._current} / {self._total} [{time_str}, {self._rate:.{self._precision}f} {self._unit}s/sec]"
                )
            else:
                print(
                    f"{self._current} / {self._total} [{time_str}, {1 / self._rate:.{self._precision}f} sec/{self._unit}]"
                )
        self._current += 1  # type: ignore (no `None` here)
        return item

    def update(self, inc: int = 1):
        """Manually update the progress meter.

        Args:
            inc (int): Number of items processed.
        """

        if not self._is_running:
            raise Exception("Progress is not running")
        self._current += inc  # type: ignore (no `None` here)
        self._current_time = time.time()

        elapsed = self._current_time - self._initial_time  # type: ignore (no `None` here)
        self._rate = (self._current - self._initial) / elapsed  # type: ignore (no `None` here)

        if self._logger is not None:
            self._old_length = self._logger.log_progress()
        else:
            left = (
                (self._total - self._current) / self._rate if self._rate > 0.0 else 0.0  # type: ignore (no `None` here)
            )
            estimated = elapsed if left < 0.0 else elapsed + left
            time_str = f"{int(elapsed // 60)}:{elapsed % 60:.0f}<{int(estimated // 60)}:{estimated % 60:.0f}"
            if self._rate >= 1.0 or self._rate == 0.0:
                print(
                    f"{self._current} / {self._total} [{time_str}, {self._rate:.{self._precision}f} {self._unit}s/sec]"
                )
            else:
                print(
                    f"{self._current} / {self._total} [{time_str}, {1 / self._rate:.{self._precision}f} sec/{self._unit}]"
                )

    def get_current(self):
        """Get the current progress.

        Returns:
            int: Current progress.
        """

        if not self._is_active:
            raise Exception("Progress is not active")
        return self._current


class Logger:
    """Logger class for logging messages and progress updates."""

    quiet: bool
    _file = sys.stderr

    progress: Progress
    scope: str | None
    _prefix: str

    def __init__(
        self,
        *,
        scope: str | None = None,
        quiet: bool = False,
        unit: str | None = None,
        precision: int | None = None,
        file=None,
    ):
        """Initialize the Logger class.

        Args:
            scope (str | None, optional): Scope of the logger. Defaults to `None`.
            quiet (bool, optional): Whether to suppress logging. Defaults to `False`.
            unit (str | None, optional): Unit for progress. Defaults to `Progress`'s default unit.
            precision (int | None, optional): Precision for progress. Defaults to `Progress`'s default precision.
            file (file-like object, optional): File-like object to write to. Defaults to `sys.stderr`.
        """
        self.quiet = quiet
        self.scope = scope
        if scope is None or not isinstance(scope, str):
            self._prefix = ""
        else:
            self._prefix = scope + "::"
        if file is None:
            file = sys.stderr
        self._file = file
        self.progress = Progress(unit=unit, logger=self, precision=precision)

    def info(self, msg: str):
        """Logs an info message.

        Args:
            msg (str): Message to log.
        """

        if self.quiet:
            return
        if not isinstance(msg, str):
            raise Exception("`msg` should be string")
        self._file.write(f"[{self._prefix}INFO]: {msg}\n")
        getattr(self._file, "flush", lambda: None)

    def debug(self, msg: str):
        """Logs a debug message.

        Args:
            msg (str): Message to log.
        """

        if self.quiet:
            return
        if not isinstance(msg, str):
            raise Exception("`msg` should be string")
        self._file.write(f"[{self._prefix}DEBUG]: {msg}\n")
        getattr(self._file, "flush", lambda: None)

    def warn(self, msg: str):
        """Logs a warning message.

        Args:
            msg (str): Message to log.
        """

        if self.quiet:
            return
        if not isinstance(msg, str):
            raise Exception("`msg` should be string")
        self._file.write(f"[{self._prefix}WARN]: {msg}\n")
        getattr(self._file, "flush", lambda: None)

    def error(self, msg: str):
        """Logs an error message.

        Args:
            msg (str): Message to log.
        """

        if self.quiet:
            return
        if not isinstance(msg, str):
            raise Exception("`msg` should be string")
        self._file.write(f"[{self._prefix}ERROR]: {msg}\n")
        getattr(self._file, "flush", lambda: None)

    def log_progress(self):
        """Logs progress information.

        Note: The method is called automatically when the progress is updated.
        """

        if self.quiet:
            return

        elapsed = self.progress._current_time - self.progress._initial_time  # type: ignore (no `None` here)
        left = (
            (self.progress._total - self.progress._current) / self.progress._rate  # type: ignore (no `None` here)
            if self.progress._rate > 0.0  # type: ignore (no `None` here)
            else 0.0
        )
        estimated = elapsed if left < 0.0 else elapsed + left
        time_str = f"{int(elapsed // 60)}:{elapsed % 60:02.0f}<{int(estimated // 60)}:{estimated % 60:02.0f}"

        if self.progress._rate >= 1.0 or self.progress._rate == 0.0:  # type: ignore (no `None` here)
            msg = f"{self.progress._current} / {self.progress._total} [{time_str}, {self.progress._rate:.{self.progress._precision}f} {self.progress._unit}s/sec]"
        else:
            msg = f"{self.progress._current} / {self.progress._total} [{time_str}, {1 / self.progress._rate:.{self.progress._precision}f} sec/{self.progress._unit}]"  # type: ignore (no `None` here)
        msg = f"\r[{self._prefix}PROGRESS]: {msg}"
        if self.progress._old_length is None:
            self._file.write(msg)
        else:
            self._file.write(msg.ljust(self.progress._old_length))
        if self.progress._current >= self.progress._total:  # type: ignore (no `None` here)
            self._file.write("\n")
        getattr(self._file, "flush", lambda: None)
        return len(msg)
