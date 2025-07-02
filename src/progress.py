class ProgressBar:
    """
    A simple terminal-based progress bar for tracking iterative processes.

    This class prints a visual progress bar to the terminal, useful for loops or tasks
    where progress feedback is desired. The progress bar can be customized with various
    display options such as prefix/suffix text, bar width, symbols, and decimal precision.
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 1,
        size: int = 50,
        fill: str = "█",
        empty: str = "-",
        line_end: str = "\r",
    ):
        """
        Initializes a new ProgressBar instance.

        Parameters:
            total (int): Total number of iterations to track.
            prefix (str, optional): Optional prefix string. Defaults to "".
            suffix (str, optional): Optional suffix string. Defaults to "".
            decimals (int, optional): Positive number of decimals in percent complete. Defaults to 1.
            size (int, optional): Character length of the progress bar. Defaults to 50.
            fill (str, optional): Bar fill character. Defaults to "█".
            empty (str, optional): Bar empty character. Defaults to "-".
            line_end (str, optional): End character (e.g., "\r" to overwrite line, "\n" to newline). Defaults to "\r".
        """
        self._total = total
        self._prefix = prefix
        self._suffix = suffix
        self._decimals = decimals
        self._size = size
        self._fill = fill
        self._empty = empty
        self._end = line_end

    def update(
        self,
        iteration: int,
    ):
        """
        Updates the progress bar display to reflect the current iteration.

        Parameters:
            iteration (int): Current iteration number (starting from 0 up to total).
        """
        if self._prefix == "":
            prefix = f"{iteration}/{self._total}"

        percent = ("{0:." + str(self._decimals) + "f}").format(
            100 * (iteration / float(self._total))
        )
        filled_length = int(self._size * iteration // self._total)
        bar_text = (self._fill * filled_length) + self._empty * (
            self._size - filled_length
        )

        print(f"\r{prefix} |{bar_text}| {percent}% {self._suffix}", end=self._end)

        # Print New Line on Complete
        if iteration == self._total:
            print()
