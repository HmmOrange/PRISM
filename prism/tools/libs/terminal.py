import os
import re
import asyncio
import platform
from typing import Optional
from asyncio import Queue
from asyncio.subprocess import PIPE, STDOUT

from utils.logs import logger
from utils.constants import DEFAULT_WORKSPACE_ROOT

from prism.utils.report import END_MARKER_VALUE, TerminalReporter
from prism.tools.tool_registry import register_tool

@register_tool()
class Terminal:
    """
    A tool for running terminal commands.
    Don't initialize a new instance of this class if one already exists.
    For commands that need to be executed within a Conda environment, it is recommended
    to use the `execute_in_conda_env` method.
    """
    
    def __init__(self):
        # Detect platform and set appropriate shell command
        if platform.system() == "Windows":
            self.shell_command = ["cmd", "/c"]
            self.command_terminator = "\r\n"
        else:
            self.shell_command = ["bash"]
            self.command_terminator = "\n"
            
        self.stdout_queue = Queue(maxsize=1000)
        self.observer = TerminalReporter()
        self.process: Optional[asyncio.subprocess.Process] = None
        #  The cmd in forbidden_terminal_commands will be replace by pass ana return the advise. example:{"cmd":"forbidden_reason/advice"}
        self.forbidden_commands = {
            "run dev": "Use Deployer.deploy_to_public instead.",
            # serve cmd have a space behind it,
            "serve ": "Use Deployer.deploy_to_public instead.",
        }

    async def _start_process(self):
        # Start a persistent shell process
        try:
            if platform.system() == "Windows":
                # Use cmd on Windows
                self.process = await asyncio.create_subprocess_exec(
                    "cmd",
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=STDOUT,
                    env=os.environ.copy(),
                    cwd=DEFAULT_WORKSPACE_ROOT.absolute(),
                )
            else:
                # Use bash on Unix-like systems
                self.process = await asyncio.create_subprocess_exec(
                    *self.shell_command,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=STDOUT,
                    executable="bash",
                    env=os.environ.copy(),
                    cwd=DEFAULT_WORKSPACE_ROOT.absolute(),
                )
            await self._check_state()
        except NotImplementedError:
            # Fallback for Windows when asyncio subprocess is not supported
            if platform.system() == "Windows":
                logger.warning("asyncio subprocess not supported on Windows, using synchronous subprocess")
                self.process = None
                # We'll handle this in run_command
            else:
                raise

    async def _check_state(self):
        """
        Check the state of the terminal, e.g. the current directory of the terminal process. Useful for agent to understand.
        """
        if self.process is not None:
            output = await self.run_command("cd" if platform.system() == "Windows" else "pwd")
            logger.info("The terminal is at:", output)

    async def run_command(self, cmd: str, daemon=False) -> str:
        """
        Executes a specified command in the terminal and streams the output back in real time.
        This command maintains state across executions, such as the current directory,
        allowing for sequential commands to be contextually aware.

        Args:
            cmd (str): The command to execute in the terminal.
            daemon (bool): If True, executes the command in an asynchronous task, allowing
                           the main program to continue execution.
        Returns:
            str: The command's output or an empty string if `daemon` is True. Remember that
                 when `daemon` is True, use the `get_stdout_output` method to get the output.
        """
        if self.process is None:
            await self._start_process()

        output = ""
        # Remove forbidden commands
        commands = re.split(r"\s*&&\s*", cmd)
        for cmd_name, reason in self.forbidden_commands.items():
            # "true" is a pass command in linux terminal.
            for index, command in enumerate(commands):
                if cmd_name in command:
                    output += f"Failed to execut {command}. {reason}\n"
                    commands[index] = "true" if platform.system() != "Windows" else "echo."
        cmd = " && ".join(commands)

        # Handle Windows fallback
        if self.process is None and platform.system() == "Windows":
            return await self._run_command_sync(cmd, daemon)

        # Send the command
        self.process.stdin.write((cmd + self.command_terminator).encode())
        self.process.stdin.write(
            f'echo "{END_MARKER_VALUE}"{self.command_terminator}'.encode()  # write EOF
        )  # Unique marker to signal command end
        await self.process.stdin.drain()
        if daemon:
            asyncio.create_task(self._read_and_process_output(cmd))
        else:
            output += await self._read_and_process_output(cmd)

        return output

    async def _run_command_sync(self, cmd: str, daemon=False) -> str:
        """
        Fallback method for Windows when asyncio subprocess is not supported
        """
        import subprocess
        
        try:
            if daemon:
                # For daemon mode, start in background
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=DEFAULT_WORKSPACE_ROOT.absolute(),
                    env=os.environ.copy()
                )
                # Store process reference for later cleanup
                self._background_process = process
                return ""
            else:
                # For synchronous execution
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=DEFAULT_WORKSPACE_ROOT.absolute(),
                    env=os.environ.copy()
                )
                return result.stdout
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return f"Error: {str(e)}"

    async def execute_in_conda_env(self, cmd: str, env, daemon=False) -> str:
        """
        Executes a given command within a specified Conda environment automatically without
        the need for manual activation. Users just need to provide the name of the Conda
        environment and the command to execute.

        Args:
            cmd (str): The command to execute within the Conda environment.
            env (str, optional): The name of the Conda environment to activate before executing the command.
                                 If not specified, the command will run in the current active environment.
            daemon (bool): If True, the command is run in an asynchronous task, similar to `run_command`,
                           affecting error logging and handling in the same manner.

        Returns:
            str: The command's output, or an empty string if `daemon` is True, with output processed
                 asynchronously in that case.

        Note:
            This function wraps `run_command`, prepending the necessary Conda activation commands
            to ensure the specified environment is active for the command's execution.
        """
        cmd = f"conda run -n {env} {cmd}"
        return await self.run_command(cmd, daemon=daemon)

    async def get_stdout_output(self) -> str:
        """
        Retrieves all collected output from background running commands and returns it as a string.

        Returns:
            str: The collected output from background running commands, returned as a string.
        """
        output_lines = []
        while not self.stdout_queue.empty():
            line = await self.stdout_queue.get()
            output_lines.append(line)
        return "\n".join(output_lines)

    async def _read_and_process_output(self, cmd, daemon=False) -> str:
        if self.process is None:
            return ""
            
        async with self.observer as observer:
            cmd_output = []
            await observer.async_report(cmd + self.command_terminator, "cmd")
            # report the command
            # Read the output until the unique marker is found.
            # We read bytes directly from stdout instead of text because when reading text,
            # '\r' is changed to '\n', resulting in excessive output.
            tmp = b""
            while True:
                output = tmp + await self.process.stdout.read(1)
                if not output:
                    continue
                *lines, tmp = output.splitlines(True)
                for line in lines:
                    line = line.decode()
                    ix = line.rfind(END_MARKER_VALUE)
                    if ix >= 0:
                        line = line[0:ix]
                        if line:
                            await observer.async_report(line, "output")
                            # report stdout in real-time
                            cmd_output.append(line)
                        return "".join(cmd_output)
                    # log stdout in real-time
                    await observer.async_report(line, "output")
                    cmd_output.append(line)
                    if daemon:
                        await self.stdout_queue.put(line)

    async def close(self):
        """Close the persistent shell process."""
        if self.process is not None:
            try:
                # Type assertion to help linter understand process is not None
                process = self.process
                process.stdin.close()
                await process.wait()
            except Exception:
                pass
        
        # Clean up background processes on Windows
        if hasattr(self, '_background_process') and self._background_process:
            try:
                self._background_process.terminate()
                self._background_process.wait()
            except Exception:
                pass