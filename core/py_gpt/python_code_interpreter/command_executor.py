import subprocess
import threading

from loguru import logger

from smart_base_model.messaging.behavior_subject import BehaviorSubject


class CommandExecutor:
    """
    The `CommandExecutor` class is responsible for executing external commands, capturing their output, and managing the execution process. It provides the following functionality:

    - Initializes a subprocess using `subprocess.Popen` and captures the stdout and stderr streams.
    - Provides an `execute` method to run the command, with an option to run it asynchronously.
    - Flushes the output streams and logs the output to a `BehaviorSubject` instance.
    - Handles exceptions that may occur during command execution and logs them.
    - Provides a method to kill the currently running process.

    The `CommandExecutor` class is designed to be used as part of a larger system that needs to execute external commands and process their output.
    """

    def __init__(self, log_behavior_subject: BehaviorSubject[str]) -> None:
        self.process = None
        self.should_kill = False
        self.is_executing = True
        self.log_behavior_subject = log_behavior_subject
        self.exception_signal = BehaviorSubject[str]()
        self.stdout_queue: list[str] = []
        self.stderr_queue: list[str] = []

    def _init_popen(self, command: str) -> bool:
        try:
            self.is_executing = True
            self.process = subprocess.Popen(
                command.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self._flush_output()
            return True
        except Exception as error:
            logger.exception(error)
            return False
        finally:
            self.is_executing = False

    def execute(self, command: str, is_async_execution: bool = True) -> bool:
        func = lambda: self._init_popen(command)  # noqa: E731
        if self.process is not None:
            self.process.kill()
            return False
        if is_async_execution:
            threading.Thread(target=func).start()
            return True
        else:
            return func()

    def clear_queues(self) -> None:
        self.stdout_queue.clear()
        self.stderr_queue.clear()

    def _flush_output(self) -> None:
        if self.process is None:
            return

        if self.process.stdout is not None:
            for line in self.process.stdout:
                line = line.strip()
                self.log_behavior_subject.next(line)
                self.stdout_queue.append(line)
                self._handle_kill_process()
        if self.process.stderr is not None:
            error = ""
            for line in self.process.stderr:
                self.stderr_queue.append(line)
                self.log_behavior_subject.next(line)
                line = line.strip()
                error += line

            if len(error) != 0:
                logger.critical(error)

            self.log_behavior_subject.next(error)
            self.exception_signal.next(error)
            self._handle_kill_process()

    def kill_current_process(self) -> None:
        self.should_kill = True

    def _handle_kill_process(self) -> None:
        if self.process is None:
            return

        if not self.should_kill:
            return

        self.process.terminate()
        self.should_kill = False


if __name__ == "__main__":
    subject = BehaviorSubject[str]()
    executor = CommandExecutor(subject)
    subject.subscribe(print)

    executor.execute("ls", is_async_execution=False)
