import sys
import logging
import os
import csv


class DummyLogger:
    def __init__(self):
        pass

    def _noop(self, *args, **kwargs):
        pass

    info = warning = debug = _noop


def create_logger(level='INFO', rank=0):
    if rank == 0:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        veb = logging.INFO
        level = str(level).upper()
        if level == 'DEBUG':
            veb = logging.DEBUG
        logger.setLevel(veb)
        logger.addHandler(handler)

        return logger
    else:
        return DummyLogger()


class CsvWriter:
    """A logging object writing to a CSV file."""

    def __init__(self, fname: str) -> None:
        """Args:
        fname: File name(path) for file to be written to.
        """
        if fname is not None and fname != '':
            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

        self._fname = fname
        self._header_written = False
        self._headers = None

        self._check_has_content()

    def _check_has_content(self):
        if not self._header_written and os.path.exists(self._fname):
            with open(self._fname, 'r', encoding='utf8') as csv_file:
                content = csv.reader(csv_file)
                rows = list(content)
                if len(rows) > 0:
                    self._header_written = True
                    self._headers = rows[0]

    def write(self, values: dict) -> None:
        """Appends given values as new row to CSV file."""
        if self._fname is None or self._fname == '':
            return

        if self._headers is None:
            self._headers = sorted(values.keys())

        # Open a file in 'append' mode, so we can continue logging safely to the
        # same file after e.g. restarting from a checkpoint.
        with open(self._fname, 'a', encoding='utf8') as file_:
            # Always use same fieldnames to create writer, this way a consistency
            # check is performed automatically on each write.
            writer = csv.DictWriter(file_, fieldnames=self._headers)
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(values)

    def close(self) -> None:
        """Closes the `CsvWriter`."""
        pass
