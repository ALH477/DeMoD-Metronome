import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import argparse
import sys
import time
import pygame  # For optional audio and Pygame renderer

class DeMoDMetronome:
    """
    Professional, robust implementation of the DeMoD Metronome system.
    Generates Sierpinski triangle fractal using seed-controlled randomness, synchronized to a metronome.
    Supports interactive events, color changes, pause/resume, BPM-based metronome mode with simple clicker audio beeps.
    Now with customizable triangle vertices, gradient coloring option, improved performance, and optional Pygame renderer with V-sync.
    Includes a pure metronome mode for simple clicker without visualization.
    """
    
    def __init__(self, seed=42, n_points=25000, batch_size=60, point_color='#0066ff',
                 bpm=None, interactive=False, save=False, verbose=False,
                 vertices=None, gradient=False, audio=False, renderer='matplotlib',
                 pure_metronome=False, beep_frequency=440):
        """
        Initialize the metronome with parameters.
        
        Args:
            seed (int): Random seed for reproducible generation.
            n_points (int): Total points to generate.
            batch_size (int): Points added per animation frame.
            point_color (str): Initial color for points (hex or named color). Ignored if gradient=True.
            bpm (int, optional): Beats Per Minute for metronome mode (frames per minute).
            interactive (bool): Enable keypress events for reactivity.
            save (bool): Save as MP4 instead of showing live (only for matplotlib renderer).
            verbose (bool): Print detailed progress.
            vertices (list of tuples, optional): Custom triangle vertices as [(x1,y1), (x2,y2), (x3,y3)].
            gradient (bool): Enable color gradient based on iteration (from blue to red).
            audio (bool): Enable simple clicker audio beeps for each beat in metronome mode (requires BPM set).
            renderer (str): Renderer backend: 'matplotlib' or 'pygame' (for V-sync support).
            pure_metronome (bool): Run in pure metronome mode (audio clicker only, no visualization).
            beep_frequency (int): Frequency of the clicker beep in Hz (default: 440).
        """
        if n_points < 1:
            raise ValueError("n_points must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if bpm is not None and (bpm < 1 or bpm > 1000):
            raise ValueError("bpm must be between 1 and 1000 if set")
        if audio and bpm is None:
            raise ValueError("audio requires bpm to be set")
        if renderer not in ['matplotlib', 'pygame']:
            raise ValueError("renderer must be 'matplotlib' or 'pygame'")
        if save and renderer == 'pygame':
            raise ValueError("save mode is only supported for matplotlib renderer")
        if pure_metronome and not bpm:
            raise ValueError("pure_metronome requires bpm to be set")
        
        self.seed = seed
        self.n_points = n_points
        self.batch_size = batch_size
        self.point_color = point_color
        self.bpm = bpm
        self.interactive = interactive
        self.save = save
        self.verbose = verbose
        self.gradient = gradient
        self.audio = audio
        self.renderer = renderer
        self.pure_metronome = pure_metronome
        self.beep_frequency = beep_frequency
        
        # Calculate interval: default 20ms (~50 FPS), or BPM-based
        if self.bpm:
            self.interval = 60000 / self.bpm  # ms per frame for BPM frames/min
            if self.verbose:
                print(f"Metronome mode: {self.bpm} BPM → interval={self.interval:.1f}ms/beat")
        else:
            self.interval = 20
        
        if not self.pure_metronome:
            # Vertices: custom or default equilateral
            if vertices:
                if len(vertices) != 3 or not all(len(v) == 2 for v in vertices):
                    raise ValueError("vertices must be a list of 3 (x,y) tuples")
                self.vertices = np.array(vertices, dtype=float)
            else:
                self.height = np.sqrt(3) / 2
                self.vertices = np.array([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, self.height]
                ], dtype=float)
            
            # Generate all points upfront (seed-controlled)
            if self.verbose:
                print(f"Generating {self.n_points:,} points with seed {self.seed}...")
            self.points = self._generate_chaos_points()
            if self.verbose:
                print("Points generated successfully.")
            
            # For gradient: precompute colors if enabled
            if self.gradient:
                self.colors = self._generate_gradient_colors()
            else:
                self.colors = None
        
        # Animation state
        self.paused = False
        self.current_frame = 0
        self.color_options = ['#0066ff', '#ff6600', '#00ff66', '#ff00ff', '#ffff00']
        self.current_color_idx = 0
        
        # Audio setup if enabled
        self.beep_sound = None
        if self.audio or (self.renderer == 'pygame' and not self.pure_metronome):
            try:
                pygame.mixer.init()
                if self.audio:
                    self.beep_sound = self._generate_beep_sound()
            except pygame.error as e:
                print(f"Warning: Audio initialization failed: {e}. Disabling audio.")
                self.audio = False
        
        if not self.pure_metronome:
            # Renderer-specific setup
            if self.renderer == 'matplotlib':
                self.fig, self.ax = plt.subplots(figsize=(10, 10 * np.ptp(self.vertices[:,1]) / np.ptp(self.vertices[:,0])))
                self._setup_matplotlib_plot()
                if self.interactive:
                    self.fig.canvas.mpl_connect('key_press_event', self._on_matplotlib_key_press)
                    if self.verbose:
                        print("Interactive mode enabled (Matplotlib). Keys: p=Pause/Resume, c=Change Color, g=Toggle Gradient, q=Quit")
            else:  # pygame
                if self.verbose:
                    print("Using Pygame renderer with V-sync enabled")
                if self.interactive and self.verbose:
                    print("Interactive mode enabled (Pygame). Keys: p=Pause/Resume, c=Change Color, g=Toggle Gradient, f=Toggle Fullscreen, q/Esc=Quit")
    
    def _generate_chaos_points(self):
        """Generate points using Chaos Game rules with seed reproducibility."""
        random.seed(self.seed)
        points = np.empty((self.n_points, 2), dtype=float)
        # Random start inside triangle
        current = np.array(self._get_random_point_in_triangle())
        points[0] = current
        
        for i in range(1, self.n_points):
            v = random.choice(self.vertices)
            current = (current + v) / 2
            points[i] = current
        return points
    
    def _get_random_point_in_triangle(self):
        """Barycentric coordinates for random point inside triangle."""
        v1, v2, v3 = self.vertices
        r1 = random.random()
        r2 = random.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        r3 = 1 - r1 - r2
        x = r1 * v1[0] + r2 * v2[0] + r3 * v3[0]
        y = r1 * v1[1] + r2 * v2[1] + r3 * v3[1]
        return (x, y)
    
    def _generate_gradient_colors(self):
        """Precompute RGBA colors for gradient from blue to red based on iteration."""
        t = np.linspace(0, 1, self.n_points)
        r = t
        g = np.zeros_like(t)
        b = 1 - t
        a = np.full_like(t, 0.8)
        return np.column_stack((r, g, b, a))
    
    def _generate_beep_sound(self):
        """Generate a simple beep sound for the clicker using pygame."""
        frequency = self.beep_frequency
        duration = 100   # ms
        sample_rate = 44100
        n_samples = int(sample_rate * duration / 1000)
        t = np.arange(n_samples) / sample_rate
        sine_wave = np.sin(2 * np.pi * frequency * t)
        sound_array = (sine_wave * 32767).astype(np.int16)  # 16-bit
        return pygame.sndarray.make_sound(np.column_stack((sound_array, sound_array)))  # Stereo
    
    def _setup_matplotlib_plot(self):
        """Initialize Matplotlib plot elements."""
        x_min, x_max = self.vertices[:,0].min() - 0.05, self.vertices[:,0].max() + 0.05
        y_min, y_max = self.vertices[:,1].min() - 0.05, self.vertices[:,1].max() + 0.05
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Plot corners
        self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], color='red', s=150,
                        zorder=5, edgecolors='darkred', linewidth=2, label='Original Corners')
        
        # Dynamic scatter for points
        self.scat = self.ax.scatter([], [], s=0.9, alpha=0.8)
        
        # Title (updated dynamically)
        self.ax.set_title(self._get_title_text(), fontsize=15, pad=25, fontweight='bold')
        
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
    
    def _get_title_text(self, end_idx=None):
        """Get title string with current status."""
        if end_idx is None:
            end_idx = min(self.current_frame * self.batch_size, self.n_points)
        mode = f" | Metronome: {self.bpm} BPM{' + Clicker' if self.audio else ''}" if self.bpm else ""
        grad = " | Gradient: On" if self.gradient else ""
        status = " (Paused)" if self.paused else ""
        rend = f" | Renderer: {self.renderer.capitalize()}"
        return f'DeMoD Metronome → Sierpinski Triangle\nSeed: {self.seed} | {end_idx:,} / {self.n_points:,} points{status}{mode}{grad}{rend}'
    
    def _init_matplotlib_animation(self):
        """Matplotlib animation init function."""
        self.scat.set_offsets(np.empty((0, 2)))
        self.scat.set_facecolors(np.empty((0, 4)))
        self.current_frame = 0
        return self.scat,
    
    def _update_matplotlib_animation(self, frame):
        """Matplotlib animation update: add batch of points if not paused."""
        if self.paused:
            time.sleep(self.interval / 1000)  # Maintain timing roughly
            return self.scat,
        
        self.current_frame += 1
        start_idx = (self.current_frame - 1) * self.batch_size
        end_idx = min(self.current_frame * self.batch_size, self.n_points)
        
        # Incremental append
        new_offsets = self.points[start_idx:end_idx]
        all_offsets = np.vstack((self.scat.get_offsets(), new_offsets))
        self.scat.set_offsets(all_offsets)
        
        if self.gradient:
            new_colors = self.colors[start_idx:end_idx]
            all_colors = np.vstack((self.scat.get_facecolors(), new_colors))
            self.scat.set_facecolors(all_colors)
        else:
            self.scat.set_color(self.color_options[self.current_color_idx])
        
        self.ax.set_title(self._get_title_text(end_idx), fontsize=15, pad=25, fontweight='bold')
        
        # Play clicker beep if audio enabled
        if self.audio and self.beep_sound:
            self.beep_sound.play()
        
        return self.scat,
    
    def _on_matplotlib_key_press(self, event):
        """Handle key press events for Matplotlib reactivity."""
        if event.key == 'p':
            self.paused = not self.paused
        elif event.key == 'c' and not self.gradient:
            self.current_color_idx = (self.current_color_idx + 1) % len(self.color_options)
        elif event.key == 'g':
            self.gradient = not self.gradient
            if self.gradient and self.colors is None:
                self.colors = self._generate_gradient_colors()
        elif event.key == 'q':
            plt.close(self.fig)
            return
        
        # Update display
        end_idx = min(self.current_frame * self.batch_size, self.n_points)
        if self.gradient:
            self.scat.set_facecolors(self.colors[:end_idx])
        else:
            self.scat.set_color(self.color_options[self.current_color_idx])
        self.ax.set_title(self._get_title_text(end_idx), fontsize=15, pad=25, fontweight='bold')
        plt.draw()
    
    def _pygame_run(self):
        """Run the animation using Pygame with V-sync and robust loop."""
        try:
            pygame.init()
            info = pygame.display.Info()
            screen_width, screen_height = info.current_w, info.current_h
            window_size = (800, 800)  # Default windowed
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
            screen = pygame.display.set_mode(window_size, flags=flags, vsync=1)
            pygame.display.set_caption('DeMoD Metronome - Pygame Renderer')
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)
            fullscreen = False
            
            # Precompute scaled points and vertices
            self._pygame_rescale(screen, fullscreen)
            
            current_end_idx = 0
            running = True
            fps = 1000 / self.interval
            
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(event.size, flags=flags, vsync=1)
                        self._pygame_rescale(screen, fullscreen)
                    elif event.type == pygame.KEYDOWN and self.interactive:
                        if event.key == pygame.K_p:
                            self.paused = not self.paused
                        elif event.key == pygame.K_c and not self.gradient:
                            self.current_color_idx = (self.current_color_idx + 1) % len(self.color_options)
                        elif event.key == pygame.K_g:
                            self.gradient = not self.gradient
                            if self.gradient and self.colors is None:
                                self.colors = self._generate_gradient_colors()
                        elif event.key == pygame.K_f:
                            fullscreen = not fullscreen
                            if fullscreen:
                                screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF, vsync=1)
                            else:
                                screen = pygame.display.set_mode(window_size, flags=flags, vsync=1)
                            self._pygame_rescale(screen, fullscreen)
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                
                if not self.paused:
                    current_end_idx = min(current_end_idx + self.batch_size, self.n_points)
                    if self.audio and self.beep_sound:
                        self.beep_sound.play()
                
                # Draw background
                screen.fill((255, 255, 255))
                
                # Draw title (multi-line)
                title_lines = self._get_title_text(current_end_idx).split('\n')
                y_pos = 10
                for line in title_lines:
                    title_text = font.render(line, True, (0, 0, 0))
                    screen.blit(title_text, (10, y_pos))
                    y_pos += title_text.get_height()
                
                # Draw corners
                for vx, vy in self.scaled_vertices:
                    pygame.draw.circle(screen, (255, 0, 0), (int(vx), int(vy)), 5)
                
                # Draw points
                for i in range(current_end_idx):
                    px, py = self.scaled_points[i]
                    if self.gradient:
                        color = tuple(int(c * 255) for c in self.colors[i][:3])
                    else:
                        color = pygame.Color(self.color_options[self.current_color_idx])
                    pygame.draw.circle(screen, color, (int(px), int(py)), 1)
                
                pygame.display.flip()
                clock.tick(fps)  # Limit FPS and respect V-sync
            
        except Exception as e:
            print(f"Pygame error: {e}")
        finally:
            pygame.quit()
    
    def _pygame_rescale(self, screen, fullscreen):
        """Rescale points and vertices for Pygame screen size, with y-flip."""
        width, height = screen.get_size()
        x_min, x_max = self.vertices[:,0].min(), self.vertices[:,0].max()
        y_min, y_max = self.vertices[:,1].min(), self.vertices[:,1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        scale = min(width / x_range, height / y_range) * 0.9
        offset_x = (width - x_range * scale) / 2 - x_min * scale
        offset_y = (height - y_range * scale) / 2 - y_min * scale
        
        self.scaled_vertices = (self.vertices * scale) + [offset_x, offset_y]
        self.scaled_points = (self.points * scale) + [offset_x, offset_y]
        
        # Flip y for Pygame (origin top-left)
        self.scaled_vertices[:,1] = height - self.scaled_vertices[:,1]
        self.scaled_points[:,1] = height - self.scaled_points[:,1]
    
    def _pure_metronome_run(self):
        """Run in pure metronome mode: simple clicker audio loop."""
        if self.verbose:
            print(f"Running pure metronome at {self.bpm} BPM with clicker frequency {self.beep_frequency} Hz. Press Ctrl+C to stop.")
        try:
            pygame.init()
            pygame.mixer.init()
            beep = self._generate_beep_sound()
            while True:
                beep.play()
                time.sleep(self.interval / 1000)
        except KeyboardInterrupt:
            if self.verbose:
                print("Pure metronome stopped.")
        finally:
            pygame.mixer.quit()
            pygame.quit()
    
    def run(self):
        """Run the metronome: pure mode, live animation, or save to file."""
        try:
            if self.pure_metronome:
                self._pure_metronome_run()
            elif self.renderer == 'pygame':
                self._pygame_run()
            else:
                n_frames = (self.n_points // self.batch_size) + 10  # Extra frames for stability
                self.ani = FuncAnimation(
                    self.fig, self._update_matplotlib_animation, frames=n_frames,
                    init_func=self._init_matplotlib_animation, blit=True,
                    interval=self.interval, repeat=True
                )
                
                if self.save:
                    filename = f'demod_metronome_sierpinski_seed_{self.seed}.mp4'
                    if self.verbose:
                        print(f"Saving animation to {filename} (may take time)...")
                    self.ani.save(filename, writer='ffmpeg', fps=1000/self.interval, dpi=200,
                                  extra_args=['-vcodec', 'libx264', '-preset', 'slow', '-crf', '18'])
                    if self.verbose:
                        print("✅ Animation saved successfully!")
                else:
                    if self.verbose:
                        print("Starting live animation... (close window to exit)")
                    plt.show()
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            if self.audio:
                pygame.mixer.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DeMoD Metronome - Sierpinski Triangle Visualization with Metronome Sync',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible generation')
    parser.add_argument('--points', type=int, default=25000,
                        help='Total number of points to generate')
    parser.add_argument('--batch', type=int, default=60,
                        help='Points added per animation frame')
    parser.add_argument('--color', type=str, default='#0066ff',
                        help='Initial point color (hex or named)')
    parser.add_argument('--bpm', type=int, default=None,
                        help='Beats Per Minute for metronome mode (sets frame rate)')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable keypress events (p=pause, c=change color, g=toggle gradient, q=quit; f=fullscreen in Pygame)')
    parser.add_argument('--save', action='store_true',
                        help='Save as MP4 (requires ffmpeg, only for matplotlib)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--vertices', nargs=6, type=float, default=None,
                        help='Custom triangle vertices as x1 y1 x2 y2 x3 y3')
    parser.add_argument('--gradient', action='store_true',
                        help='Enable color gradient from blue to red based on iteration')
    parser.add_argument('--audio', action='store_true',
                        help='Enable simple clicker audio beeps for each beat (requires bpm)')
    parser.add_argument('--renderer', type=str, default='matplotlib',
                        help='Renderer backend: matplotlib or pygame (for V-sync)')
    parser.add_argument('--pure-metronome', action='store_true',
                        help='Run in pure metronome mode (clicker only, no visualization; requires bpm)')
    parser.add_argument('--beep-frequency', type=int, default=440,
                        help='Frequency of the clicker beep in Hz')
    
    args = parser.parse_args()
    
    # Process vertices if provided
    custom_vertices = None
    if args.vertices:
        custom_vertices = [(args.vertices[0], args.vertices[1]),
                           (args.vertices[2], args.vertices[3]),
                           (args.vertices[4], args.vertices[5])]
    
    try:
        metronome = DeMoDMetronome(
            seed=args.seed,
            n_points=args.points,
            batch_size=args.batch,
            point_color=args.color,
            bpm=args.bpm,
            interactive=args.interactive,
            save=args.save,
            verbose=args.verbose,
            vertices=custom_vertices,
            gradient=args.gradient,
            audio=args.audio,
            renderer=args.renderer,
            pure_metronome=args.pure_metronome,
            beep_frequency=args.beep_frequency
        )
        metronome.run()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)