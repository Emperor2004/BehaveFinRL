import logging
import sys
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Centralised logger that writes to stdout, a file and TensorBoard.

    Parameters
    ----------
    name: str
        Identifier used for the logger hierarchy.
    log_file: Optional[Path]
        File path to write a plain‑text log (created if not existent).
    level: int
        Logging level from :mod:`logging`.
    tb_log_dir: Optional[Path]
        Directory where TensorBoard logs will be stored. If ``None`` TensorBoard is
        disabled.
    """

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        tb_log_dir: Optional[Path] = None,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # console handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # optional file handler
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # optional TensorBoard writer
        self.tb_writer: Optional[SummaryWriter] = None
        if tb_log_dir:
            tb_log_dir = Path(tb_log_dir)
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    # ------------------------------------------------------------------
    # Logging shortcuts – each mirrors the standard logging API.
    # They also accept arbitrary scalar key/value pairs to be recorded in
    # TensorBoard for the current step.
    # ------------------------------------------------------------------
    def _log(self, level: int, msg: str, step: Optional[int] = None, **tb_scalars):
        self.logger.log(level, msg)
        if self.tb_writer is not None and tb_scalars:
            step = step if step is not None else 0
            for k, v in tb_scalars.items():
                self.tb_writer.add_scalar(k, v, step)

    def info(self, msg: str, step: Optional[int] = None, **tb_scalars):
        self._log(logging.INFO, msg, step, **tb_scalars)

    def warning(self, msg: str, step: Optional[int] = None, **tb_scalars):
        self._log(logging.WARNING, msg, step, **tb_scalars)

    def error(self, msg: str, step: Optional[int] = None, **tb_scalars):
        self._log(logging.ERROR, msg, step, **tb_scalars)

    def debug(self, msg: str, step: Optional[int] = None, **tb_scalars):
        self._log(logging.DEBUG, msg, step, **tb_scalars)

    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
