# DeMoD Metronome

A professional, robust Python-based metronome system that visualizes the Chaos Game to generate the Sierpinski Triangle fractal, synchronized to a rhythmic beat. This tool uses seed-controlled randomness for reproducible results, supports interactive events (pause, color change, gradient toggle, fullscreen in Pygame), and includes a metronome mode based on BPM for rhythmic progression with a simple integrated clicker (audio beeps).

Built with Matplotlib (default) or Pygame (for V-sync and smoother rendering), NumPy for computations, and designed for ease of use via command-line arguments. Includes a pure metronome mode for just the clicker without visualization.

## Features
- **Seed-Based Reproducibility**: Generate the exact same fractal every time with the same seed.
- **Interactive Controls**: Pause/resume ('p'), change point colors ('c'), toggle gradient ('g'), quit ('q'/Esc), fullscreen toggle ('f' in Pygame).
- **Metronome Mode**: Set BPM to control animation frame rate, adding points in a rhythmic "beat" style with integrated simple clicker (customizable audio beeps).
- **Pure Metronome Mode**: Run just the clicker audio loop without any visualization.
- **Customizable**: Adjustable number of points, batch size, initial color, custom vertices, gradient coloring, beep frequency.
- **Dual Renderers**: Matplotlib for high-quality plots and MP4 export; Pygame for V-sync, resizable window, and fullscreen.
- **Save to Video**: Export high-quality MP4 animations (requires FFmpeg, Matplotlib only).
- **Robust Error Handling**: Validates inputs, handles edge cases, large point counts, and provides verbose output.
- **Nix Support**: Reproducible dev environment via Nix flake.

## Installation

### Via Pip (Standard)
1. Clone the repository:
   ```
   git clone https://github.com/ALH477/demod-metronome.git
   cd demod-metronome
   ```

2. Install dependencies:
   ```
   pip install numpy matplotlib pygame
   ```

   For saving animations to MP4, install FFmpeg:
   - On macOS: `brew install ffmpeg`
   - On Ubuntu: `sudo apt install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Via Nix (Reproducible)
If you have Nix installed:
```
nix develop
```
This enters a shell with Python 3.12, NumPy, Matplotlib, Pygame, and FFmpeg pre-installed.

## Usage

Run the script with default settings:
```
python demod_metronome.py
```

### Command-Line Arguments
- `--seed <int>`: Random seed (default: 42)
- `--points <int>`: Total points to generate (default: 25000)
- `--batch <int>`: Points per animation frame (default: 60)
- `--color <str>`: Initial point color (hex or named, default: '#0066ff')
- `--bpm <int>`: Beats Per Minute for metronome mode (optional)
- `--interactive`: Enable keypress events
- `--save`: Save as MP4 instead of live display (requires FFmpeg, Matplotlib only)
- `--verbose`: Print detailed progress
- `--vertices <float> <float> ...`: Custom triangle vertices (6 floats: x1 y1 x2 y2 x3 y3)
- `--gradient`: Enable color gradient
- `--audio`: Enable simple clicker audio beeps (requires --bpm)
- `--renderer <str>`: 'matplotlib' or 'pygame' (default: matplotlib)
- `--pure-metronome`: Run in pure metronome mode (clicker only; requires --bpm)
- `--beep-frequency <int>`: Clicker beep frequency in Hz (default: 440)

Examples:
- Interactive with Pygame and V-sync: `python demod_metronome.py --renderer pygame --interactive`
- Metronome at 60 BPM with clicker: `python demod_metronome.py --bpm 60 --batch 1 --audio`
- Pure metronome: `python demod_metronome.py --bpm 120 --audio --pure-metronome --beep-frequency 880`
- Custom vertices: `python demod_metronome.py --vertices 0 0 1 0 0.5 0.866 --gradient`
- Save video: `python demod_metronome.py --save`

## How It Works
This implements the Chaos Game synchronized to a metronome:
1. Mark three corners of a triangle.
2. Pick a random starting point inside the triangle (seed-controlled).
3. Randomly choose a corner and plot the midpoint at each beat.
4. Repeat thousands of times to reveal the Sierpinski Triangle, with optional clicker sound per beat.

All randomness is seeded for reproducibility. The animation builds the fractal progressively, with options for real-time interaction and audio sync. Pure mode provides a simple standalone metronome clicker.

## Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Associated with DeMoD LLC (@DeMoDLLC on X). Developed by ALH477.
