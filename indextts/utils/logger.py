"""
Colorful Logger Module for IndexTTS2
Uses rich library for beautiful console output with colors and progress bars
"""

import sys
from typing import Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import print as rprint
from rich.style import Style


class ColorfulLogger:
    """Beautiful colorful logger with hierarchical display"""
    
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }

    def __init__(self, name: str = "IndexTTS2", enable_timestamp: bool = True, level: str = "INFO"):
        self.name = name
        self.console = Console()
        self.enable_timestamp = enable_timestamp
        self.level = self._normalize_level(level)
        
        # Define color styles
        self.styles = {
            'info': Style(color="cyan", bold=False),
            'success': Style(color="green", bold=True),
            'warning': Style(color="yellow", bold=True),
            'error': Style(color="red", bold=True),
            'debug': Style(color="magenta", dim=True),
            'stage': Style(color="blue", bold=True),
            'model': Style(color="green"),
            'time': Style(color="yellow"),
            'value': Style(color="cyan", bold=True),
            'key': Style(color="white", dim=True),
        }

    def _normalize_level(self, level: str) -> int:
        if level is None:
            return self.LEVELS["INFO"]
        if isinstance(level, str):
            return self.LEVELS.get(level.upper(), self.LEVELS["INFO"])
        if isinstance(level, int):
            return level
        return self.LEVELS["INFO"]

    def set_level(self, level: str):
        self.level = self._normalize_level(level)

    def _should_log(self, level_name: str) -> bool:
        required = self.LEVELS[level_name]
        return required >= self.level
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        if self.enable_timestamp:
            return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
        return ""
    
    def info(self, message: str, prefix: str = "ℹ"):
        """Print info message"""
        if not self._should_log("INFO"):
            return
        self.console.print(f"{self._get_timestamp()}[bold]{prefix} [/bold] {message}", style=self.styles['info'])
    
    def success(self, message: str, prefix: str = "✓"):
        """Print success message"""
        if not self._should_log("INFO"):
            return
        self.console.print(f"{self._get_timestamp()}[bold]{prefix} [/bold] {message}", style=self.styles['success'])
    
    def warning(self, message: str, prefix: str = "⚠"):
        """Print warning message"""
        if not self._should_log("WARNING"):
            return
        self.console.print(f"{self._get_timestamp()}[bold]{prefix} [/bold] {message}", style=self.styles['warning'])
    
    def error(self, message: str, prefix: str = "✗"):
        """Print error message"""
        if not self._should_log("ERROR"):
            return
        self.console.print(f"{self._get_timestamp()}[bold]{prefix} [/bold] {message}", style=self.styles['error'])
    
    def debug(self, message: str, prefix: str = "•"):
        """Print debug message"""
        if not self._should_log("DEBUG"):
            return
        from rich.text import Text
        # Create text with bold prefix and plain message (no markup parsing in message)
        text = Text()
        if self.enable_timestamp:
            text.append(f"{datetime.now().strftime('%H:%M:%S')} ", style="dim")
        text.append(f"{prefix} ", style="bold")
        text.append(message)  # Plain text, no markup interpretation
        self.console.print(text, style=self.styles['debug'], highlight=False)
    
    def stage(self, message: str):
        """Print stage header"""
        if not self._should_log("INFO"):
            return
        self.console.print()
        self.console.print(f"{'='*60}", style="blue")
        self.console.print(f"  {message}", style=self.styles['stage'])
        self.console.print(f"{'='*60}", style="blue")
    
    def model_loaded(self, model_name: str, checkpoint_path: str):
        """Print model loading success"""
        if not self._should_log("INFO"):
            return
        self.console.print(
            f"{self._get_timestamp()}[green]✓[/green] {model_name:20s} loaded from: [dim]{checkpoint_path}[/dim]"
        )
    
    def device_info(self, device: str, is_fp16: bool, use_cuda_kernel: bool):
        """Print device configuration info"""
        if not self._should_log("INFO"):
            return
        tree = Tree(f"[bold cyan]Device Configuration[/bold cyan]")
        tree.add(f"[green]Device:[/green] {device}")
        tree.add(f"[green]FP16:[/green] {is_fp16}")
        tree.add(f"[green]CUDA Kernel:[/green] {use_cuda_kernel}")
        self.console.print(tree)
    
    def print_dict(self, title: str, data: Dict[str, Any], level: int = 0):
        """Print dictionary with hierarchical structure"""
        if not self._should_log("INFO"):
            return
        indent = "  " * level
        if level == 0:
            self.console.print(f"\n{self._get_timestamp()}[bold]{title}[/bold]")
        
        for key, value in data.items():
            if isinstance(value, dict):
                self.console.print(f"{indent}  [cyan]{key}:[/cyan]")
                self.print_dict("", value, level + 1)
            elif isinstance(value, (list, tuple)):
                self.console.print(f"{indent}  [cyan]{key}:[/cyan] [yellow]{value}[/yellow]")
            elif isinstance(value, (int, float)):
                self.console.print(f"{indent}  [cyan]{key}:[/cyan] [yellow]{value:.4f}[/yellow]" if isinstance(value, float) else f"{indent}  [cyan]{key}:[/cyan] [yellow]{value}[/yellow]")
            else:
                self.console.print(f"{indent}  [cyan]{key}:[/cyan] [white]{value}[/white]")
    
    def print_time_stats(self, stats: Dict[str, float], total_time: float, audio_length: float):
        """Performance Statistics table disabled."""
        return
    
    def print_emotion_vector(self, emotion_data):
        """Print emotion vector(s) in a beautiful format
        
        Args:
            emotion_data: Can be either:
                - A single dict {emotion: value}
                - A list of dicts [{emotion: value}, ...]
        """
        if not self._should_log("INFO"):
            return
        from rich.columns import Columns
        
        # Normalize input to list
        if isinstance(emotion_data, dict):
            emotion_dicts = [emotion_data]
        else:
            emotion_dicts = emotion_data
        
        # Create trees for each emotion vector
        trees = []
        for idx, emotion_dict in enumerate(emotion_dicts):
            # Use different title for multiple vectors
            if len(emotion_dicts) > 1:
                title = f"[bold yellow]🎭 Emotion Vector #{idx+1}[/bold yellow]"
            else:
                title = "[bold yellow]🎭 Emotion Vector[/bold yellow]"
            
            tree = Tree(title)
            
            for emotion, value in emotion_dict.items():
                # Color based on value
                if value > 0.7:
                    color = "red"
                elif value > 0.4:
                    color = "yellow"
                elif value > 0.1:
                    color = "cyan"
                else:
                    color = "dim"
                
                # Create bar visualization
                bar_length = int(value * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                tree.add(f"[{color}]{emotion:12s}[/{color}] [{color}]{bar}[/{color}] [{color}]{value:.3f}[/{color}]")
            
            trees.append(tree)
        
        # Display side by side if multiple vectors
        if len(trees) > 1:
            self.console.print(Columns(trees, equal=True, expand=True))
        else:
            self.console.print(trees[0])
    
    def panel(self, message: str, title: str = "", style: str = "cyan"):
        """Print message in a panel"""
        if not self._should_log("INFO"):
            return
        self.console.print(Panel(message, title=title, border_style=style))
    
    def rule(self, title: str = "", style: str = "cyan"):
        """Print a horizontal rule"""
        if not self._should_log("INFO"):
            return
        from rich.rule import Rule
        self.console.print(Rule(title, style=style))
    
    def print_raw(self, *args, **kwargs):
        """Print raw message (for compatibility)"""
        self.console.print(*args, **kwargs)
    
    def print_segment_stats(self, segment_positions: Dict[int, list], total_tokens: int, selected_beam: int):
        """Print multi-segment semantic token statistics in a compact format
        
        Args:
            segment_positions: Dict of {beam_id: [switch_positions]}
            total_tokens: Total number of semantic tokens
            selected_beam: ID of the selected beam (can be int or tensor)
        """
        if not self._should_log("INFO"):
            return
        from rich.panel import Panel
        from rich.columns import Columns
        import torch
        
        # Convert tensor to int if necessary
        if isinstance(selected_beam, torch.Tensor):
            selected_beam = selected_beam.item()
        
        # Create panels for each beam
        beam_panels = []
        for beam_id in sorted(segment_positions.keys()):
            is_selected = beam_id == selected_beam
            positions = segment_positions[beam_id]
            all_positions = [0] + positions + [total_tokens]
            num_segments = len(all_positions) - 1
            
            # Build compact display for this beam
            if is_selected:
                header = f"[green]⭐ Beam {beam_id}[/green]  [cyan]Total:[/cyan] [yellow]{total_tokens}[/yellow]  [cyan]Seg:[/cyan] [yellow]{num_segments}[/yellow]"
            else:
                header = f"[dim]Beam {beam_id}[/dim]  [cyan]Total:[/cyan] [yellow]{total_tokens}[/yellow]  [cyan]Seg:[/cyan] [yellow]{num_segments}[/yellow]"
            
            content = header + "\n\n"
            
            for seg_idx in range(len(all_positions) - 1):
                start_pos = all_positions[seg_idx]
                end_pos = all_positions[seg_idx + 1]
                seg_len = end_pos - start_pos
                percentage = (seg_len / total_tokens * 100) if total_tokens > 0 else 0
                
                # Compact bar (12 chars for narrow panels)
                bar_length = int(percentage / 100 * 12)
                bar = "█" * bar_length + "░" * (12 - bar_length)
                
                # Color based on percentage
                if percentage > 40:
                    bar_color = "red"
                elif percentage > 25:
                    bar_color = "yellow"
                else:
                    bar_color = "green"
                
                # Compact single line per segment
                if is_selected:
                    content += f"[cyan]§{seg_idx + 1}[/cyan] [{bar_color}]{bar}[/{bar_color}] [yellow]{seg_len:3d}[/yellow] [{bar_color}]{percentage:4.1f}%[/{bar_color}]\n"
                else:
                    content += f"[dim][cyan]§{seg_idx + 1}[/cyan] [{bar_color}]{bar}[/{bar_color}] [yellow]{seg_len:3d}[/yellow] [{bar_color}]{percentage:4.1f}%[/{bar_color}][/dim]\n"
            
            # Create panel with different style for selected beam
            border_style = "green" if is_selected else "dim"
            panel = Panel(
                content.rstrip(), 
                border_style=border_style,
                expand=False,
                padding=(0, 1)
            )
            beam_panels.append(panel)
        
        self.console.print()
        self.console.print(Columns(beam_panels, equal=False, expand=False))
        self.console.print()


class ColorfulProgress:
    """Colorful progress bar wrapper"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.task_id = None
    
    def __enter__(self):
        self.progress.__enter__()
        return self
    
    def __exit__(self, *args):
        self.progress.__exit__(*args)
    
    def add_task(self, description: str, total: float = 100.0):
        """Add a new task"""
        self.task_id = self.progress.add_task(description, total=total)
        return self.task_id
    
    def update(self, advance: float = 0, description: str = None, completed: float = None):
        """Update progress"""
        kwargs = {}
        if advance:
            kwargs['advance'] = advance
        if description:
            kwargs['description'] = description
        if completed is not None:
            kwargs['completed'] = completed
        
        if self.task_id is not None:
            self.progress.update(self.task_id, **kwargs)


# Global logger instance
_logger_instance = None
_UNSET = object()


def get_logger(name: str = "IndexTTS2", log_level: Optional[str] = _UNSET) -> ColorfulLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        init_level = "INFO" if log_level is _UNSET else log_level
        _logger_instance = ColorfulLogger(name, level=init_level)
    else:
        if log_level is not _UNSET:
            _logger_instance.set_level(log_level)
    return _logger_instance
    return _logger_instance


def create_progress(console: Optional[Console] = None) -> ColorfulProgress:
    """Create a new progress bar"""
    return ColorfulProgress(console)
