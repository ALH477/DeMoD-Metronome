"""
DeMoD Metronome — Sierpinski Triangle fractal visualizer synchronized to a metronome.

Key improvements over v1:
  - Consistent NumPy RNG (no stdlib random mixing)
  - O(1)-per-frame Pygame rendering via persistent backing surface
  - O(n) Matplotlib rendering via local offsets buffer (no get_offsets round-trip)
  - Proper pause/resume via FuncAnimation.event_source
  - Fixed FuncAnimation repeat=False (was erasing all points on loop)
  - Fixed double pygame.quit in run() finally block
  - Type hints throughout
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
import sys
import time
import pygame
from typing import Optional


class DeMoDMetronome:
    """
    Sierpinski triangle fractal generator synchronized to a BPM metronome.
    Supports Matplotlib and Pygame renderers, gradient coloring, audio beeps,
    and interactive controls.
    """

    def __init__(
        self,
        seed: int = 42,
        n_points: int = 25000,
        batch_size: int = 60,
        point_color: str = "#0066ff",
        bpm: Optional[int] = None,
        interactive: bool = False,
        save: bool = False,
        verbose: bool = False,
        vertices: Optional[list] = None,
        gradient: bool = False,
        audio: bool = False,
        renderer: str = "matplotlib",
        pure_metronome: bool = False,
        beep_frequency: int = 440,
    ):
        # --- Validation -------------------------------------------------
        if n_points < 1:
            raise ValueError("n_points must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if bpm is not None and not (1 <= bpm <= 1000):
            raise ValueError("bpm must be between 1 and 1000")
        if audio and bpm is None:
            raise ValueError("audio requires bpm to be set")
        if renderer not in ("matplotlib", "pygame"):
            raise ValueError("renderer must be 'matplotlib' or 'pygame'")
        if save and renderer == "pygame":
            raise ValueError("save mode is only supported for the matplotlib renderer")
        if pure_metronome and not bpm:
            raise ValueError("pure_metronome requires bpm to be set")

        # --- Parameters -------------------------------------------------
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

        # BPM → frame interval
        self.interval: float = (60_000 / bpm) if bpm else 20.0  # ms
        if bpm and verbose:
            print(f"Metronome mode: {bpm} BPM → {self.interval:.1f} ms/beat")

        # --- Animation state --------------------------------------------
        self.paused = False
        self.current_frame = 0
        self.color_options = ["#0066ff", "#ff6600", "#00ff66", "#ff00ff", "#ffff00"]
        self.current_color_idx = 0

        # Internal offsets buffer (avoids O(n²) get_offsets round-trips)
        self._offsets_buf: Optional[np.ndarray] = None
        self._colors_buf: Optional[np.ndarray] = None
        self._buf_size = 0  # how many points are currently displayed

        # FuncAnimation handle (kept alive)
        self.ani: Optional[FuncAnimation] = None

        if not pure_metronome:
            self._setup_geometry(vertices)
            self._rng = np.random.default_rng(seed)
            if verbose:
                print(f"Generating {n_points:,} points with seed {seed}…")
            self.points = self._generate_chaos_points()
            if verbose:
                print("Points generated.")
            self.colors = self._generate_gradient_colors() if gradient else None

        # --- Audio ------------------------------------------------------
        self.beep_sound: Optional["pygame.mixer.Sound"] = None
        needs_mixer = audio or (renderer == "pygame" and not pure_metronome) or pure_metronome
        if needs_mixer:
            try:
                pygame.mixer.pre_init(44100, -16, 2, 512)
                pygame.mixer.init()
                if audio or pure_metronome:
                    self.beep_sound = self._generate_beep_sound()
            except pygame.error as exc:
                print(f"Warning: audio init failed ({exc}). Disabling audio.")
                self.audio = False

        if not pure_metronome:
            self._setup_renderer()

    # ------------------------------------------------------------------ #
    #  Geometry & point generation                                         #
    # ------------------------------------------------------------------ #

    def _setup_geometry(self, vertices: Optional[list]) -> None:
        if vertices is not None:
            if len(vertices) != 3 or not all(len(v) == 2 for v in vertices):
                raise ValueError("vertices must be a list of 3 (x, y) tuples")
            self.vertices = np.array(vertices, dtype=float)
        else:
            h = np.sqrt(3) / 2
            self.vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])

    def _generate_chaos_points(self) -> np.ndarray:
        """Chaos game with fully vectorised NumPy RNG — no stdlib random."""
        rng = self._rng
        # Random starting point inside triangle via barycentric coords
        r = rng.random(2)
        if r.sum() > 1:
            r = 1 - r
        start = r[0] * self.vertices[0] + r[1] * self.vertices[1] + (1 - r.sum()) * self.vertices[2]

        # Precompute all vertex choices at once
        choices = rng.integers(0, 3, size=self.n_points - 1)

        points = np.empty((self.n_points, 2))
        points[0] = start
        for i, vi in enumerate(choices, start=1):
            points[i] = (points[i - 1] + self.vertices[vi]) * 0.5
        return points

    def _generate_gradient_colors(self) -> np.ndarray:
        """Blue → red RGBA gradient across all n_points."""
        t = np.linspace(0, 1, self.n_points)
        return np.column_stack((t, np.zeros_like(t), 1 - t, np.full_like(t, 0.8)))

    # ------------------------------------------------------------------ #
    #  Audio                                                               #
    # ------------------------------------------------------------------ #

    def _generate_beep_sound(self) -> "pygame.mixer.Sound":
        sr = 44100
        dur_ms = 80
        n = int(sr * dur_ms / 1000)
        t = np.arange(n) / sr
        # Slight exponential decay for a cleaner click feel
        envelope = np.exp(-t * 40)
        wave = (np.sin(2 * np.pi * self.beep_frequency * t) * envelope * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo)

    # ------------------------------------------------------------------ #
    #  Renderer setup                                                      #
    # ------------------------------------------------------------------ #

    def _setup_renderer(self) -> None:
        if self.renderer == "matplotlib":
            aspect = np.ptp(self.vertices[:, 1]) / max(np.ptp(self.vertices[:, 0]), 1e-9)
            self.fig, self.ax = plt.subplots(figsize=(10, max(2, 10 * aspect)))
            self._setup_matplotlib_plot()
            if self.interactive:
                self.fig.canvas.mpl_connect("key_press_event", self._on_matplotlib_key_press)
                if self.verbose:
                    print("Interactive (Matplotlib): p=pause  c=color  g=gradient  q=quit")
        else:
            if self.verbose:
                print("Pygame renderer with V-sync")
            if self.interactive and self.verbose:
                print("Interactive (Pygame): p=pause  c=color  g=gradient  f=fullscreen  q/Esc=quit")

    # ------------------------------------------------------------------ #
    #  Matplotlib                                                          #
    # ------------------------------------------------------------------ #

    def _setup_matplotlib_plot(self) -> None:
        pad = 0.05
        self.ax.set_xlim(self.vertices[:, 0].min() - pad, self.vertices[:, 0].max() + pad)
        self.ax.set_ylim(self.vertices[:, 1].min() - pad, self.vertices[:, 1].max() + pad)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.scatter(
            self.vertices[:, 0], self.vertices[:, 1],
            color="red", s=150, zorder=5, edgecolors="darkred", linewidths=2, label="Corners",
        )
        self.scat = self.ax.scatter([], [], s=0.9, alpha=0.8)
        self.ax.set_title(self._get_title_text(), fontsize=15, pad=25, fontweight="bold")
        plt.legend(loc="upper right", fontsize=12)
        plt.tight_layout()

    def _get_title_text(self, end_idx: int = 0) -> str:
        mode = f" | Metronome: {self.bpm} BPM{' + Clicker' if self.audio else ''}" if self.bpm else ""
        grad = " | Gradient: On" if self.gradient else ""
        status = " (Paused)" if self.paused else ""
        rend = f" | Renderer: {self.renderer.capitalize()}"
        return (
            f"DeMoD Metronome → Sierpinski Triangle\n"
            f"Seed: {self.seed} | {end_idx:,} / {self.n_points:,} pts{status}{mode}{grad}{rend}"
        )

    def _init_matplotlib_animation(self):
        self.scat.set_offsets(np.empty((0, 2)))
        self.scat.set_facecolors(np.empty((0, 4)))
        # Reset local buffers
        self._offsets_buf = np.empty((self.n_points, 2))
        self._colors_buf = np.empty((self.n_points, 4))
        self._buf_size = 0
        self.current_frame = 0
        return (self.scat,)

    def _update_matplotlib_animation(self, frame: int):
        if self.paused:
            # Do nothing; event_source is already stopped via key handler
            return (self.scat,)

        self.current_frame += 1
        start = (self.current_frame - 1) * self.batch_size
        end = min(self.current_frame * self.batch_size, self.n_points)

        # Append into local buffer — O(batch) not O(n)
        new_count = end - start
        self._offsets_buf[self._buf_size: self._buf_size + new_count] = self.points[start:end]
        if self.gradient and self.colors is not None:
            self._colors_buf[self._buf_size: self._buf_size + new_count] = self.colors[start:end]
        self._buf_size += new_count

        self.scat.set_offsets(self._offsets_buf[: self._buf_size])

        if self.gradient and self.colors is not None:
            self.scat.set_facecolors(self._colors_buf[: self._buf_size])
        else:
            self.scat.set_color(self.color_options[self.current_color_idx])

        self.ax.set_title(self._get_title_text(self._buf_size), fontsize=15, pad=25, fontweight="bold")

        if self.audio and self.beep_sound:
            self.beep_sound.play()

        return (self.scat,)

    def _on_matplotlib_key_press(self, event) -> None:
        key = event.key
        if key == "p":
            self.paused = not self.paused
            # Properly pause/resume the animation — no sleep() in callbacks
            if self.paused:
                self.ani.event_source.stop()
            else:
                self.ani.event_source.start()
        elif key == "c" and not self.gradient:
            self.current_color_idx = (self.current_color_idx + 1) % len(self.color_options)
            self.scat.set_color(self.color_options[self.current_color_idx])
        elif key == "g":
            self.gradient = not self.gradient
            if self.gradient and self.colors is None:
                self.colors = self._generate_gradient_colors()
                self._colors_buf = np.empty((self.n_points, 4))
            if self.gradient and self.colors is not None:
                self.scat.set_facecolors(self.colors[: self._buf_size])
            else:
                self.scat.set_color(self.color_options[self.current_color_idx])
        elif key == "q":
            plt.close(self.fig)
            return

        self.ax.set_title(self._get_title_text(self._buf_size), fontsize=15, pad=25, fontweight="bold")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Pygame                                                              #
    # ------------------------------------------------------------------ #

    def _pygame_run(self) -> None:
        try:
            pygame.init()
            info = pygame.display.Info()
            screen_w, screen_h = info.current_w, info.current_h
            window_size = (800, 800)
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
            screen = pygame.display.set_mode(window_size, flags=flags, vsync=1)
            pygame.display.set_caption("DeMoD Metronome — Pygame")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)
            fullscreen = False
            fps = 1000.0 / self.interval

            self._pygame_rescale(screen)

            # Persistent backing surface — only NEW points are blitted per frame
            backing = pygame.Surface(screen.get_size(), flags=pygame.SRCALPHA)
            backing.fill((255, 255, 255))

            current_end_idx = 0
            running = True

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(event.size, flags=flags, vsync=1)
                        backing = pygame.Surface(screen.get_size(), flags=pygame.SRCALPHA)
                        backing.fill((255, 255, 255))
                        self._pygame_rescale(screen)
                        current_end_idx = 0  # redraw needed; reset point counter
                    elif event.type == pygame.KEYDOWN and self.interactive:
                        if event.key == pygame.K_p:
                            self.paused = not self.paused
                        elif event.key == pygame.K_c and not self.gradient:
                            self.current_color_idx = (self.current_color_idx + 1) % len(self.color_options)
                            # Invalidate backing so color change is visible
                            backing.fill((255, 255, 255))
                            current_end_idx = 0
                        elif event.key == pygame.K_g:
                            self.gradient = not self.gradient
                            if self.gradient and self.colors is None:
                                self.colors = self._generate_gradient_colors()
                            backing.fill((255, 255, 255))
                            current_end_idx = 0
                        elif event.key == pygame.K_f:
                            fullscreen = not fullscreen
                            if fullscreen:
                                screen = pygame.display.set_mode(
                                    (screen_w, screen_h),
                                    flags=pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF,
                                    vsync=1,
                                )
                            else:
                                screen = pygame.display.set_mode(window_size, flags=flags, vsync=1)
                            backing = pygame.Surface(screen.get_size(), flags=pygame.SRCALPHA)
                            backing.fill((255, 255, 255))
                            self._pygame_rescale(screen)
                            current_end_idx = 0
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                if not self.paused and current_end_idx < self.n_points:
                    new_end = min(current_end_idx + self.batch_size, self.n_points)

                    # Draw ONLY the new batch onto the backing surface
                    for i in range(current_end_idx, new_end):
                        px, py = int(self.scaled_points[i, 0]), int(self.scaled_points[i, 1])
                        if self.gradient and self.colors is not None:
                            color = tuple(int(c * 255) for c in self.colors[i, :3])
                        else:
                            color = pygame.Color(self.color_options[self.current_color_idx])
                        pygame.draw.circle(backing, color, (px, py), 1)

                    current_end_idx = new_end

                    if self.audio and self.beep_sound:
                        self.beep_sound.play()

                # Compose final frame: backing + UI overlay
                screen.blit(backing, (0, 0))

                # Vertex markers
                for vx, vy in self.scaled_vertices:
                    pygame.draw.circle(screen, (220, 20, 20), (int(vx), int(vy)), 5)

                # HUD
                for i, line in enumerate(self._get_title_text(current_end_idx).split("\n")):
                    surf = font.render(line, True, (0, 0, 0))
                    screen.blit(surf, (10, 10 + i * surf.get_height()))

                pygame.display.flip()
                clock.tick(fps)

        except Exception as exc:
            print(f"Pygame error: {exc}")
        finally:
            pygame.quit()

    def _pygame_rescale(self, screen: pygame.Surface) -> None:
        """Rescale points/vertices to fit screen with y-flip (Pygame origin = top-left)."""
        w, h = screen.get_size()
        x_range = np.ptp(self.vertices[:, 0]) or 1.0
        y_range = np.ptp(self.vertices[:, 1]) or 1.0
        scale = min(w / x_range, h / y_range) * 0.9
        ox = (w - x_range * scale) / 2 - self.vertices[:, 0].min() * scale
        oy = (h - y_range * scale) / 2 - self.vertices[:, 1].min() * scale

        self.scaled_vertices = self.vertices * scale + [ox, oy]
        self.scaled_points = self.points * scale + [ox, oy]

        # Flip y
        self.scaled_vertices[:, 1] = h - self.scaled_vertices[:, 1]
        self.scaled_points[:, 1] = h - self.scaled_points[:, 1]

    # ------------------------------------------------------------------ #
    #  Pure metronome                                                      #
    # ------------------------------------------------------------------ #

    def _pure_metronome_run(self) -> None:
        if self.verbose:
            print(f"Pure metronome: {self.bpm} BPM @ {self.beep_frequency} Hz. Ctrl-C to stop.")
        try:
            pygame.init()
            pygame.mixer.init()
            beep = self._generate_beep_sound()
            interval_s = self.interval / 1000.0
            while True:
                beep.play()
                time.sleep(interval_s)
        except KeyboardInterrupt:
            if self.verbose:
                print("Metronome stopped.")
        finally:
            pygame.mixer.quit()
            pygame.quit()

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Dispatch to the appropriate run mode."""
        try:
            if self.pure_metronome:
                self._pure_metronome_run()
                return

            if self.renderer == "pygame":
                self._pygame_run()
                return

            # Matplotlib path
            n_frames = (self.n_points // self.batch_size) + 10
            self.ani = FuncAnimation(
                self.fig,
                self._update_matplotlib_animation,
                frames=n_frames,
                init_func=self._init_matplotlib_animation,
                blit=True,
                interval=self.interval,
                repeat=False,  # was True — caused all points to vanish on loop
            )

            if self.save:
                fname = f"demod_metronome_sierpinski_seed_{self.seed}.mp4"
                if self.verbose:
                    print(f"Saving to {fname}…")
                self.ani.save(
                    fname, writer="ffmpeg",
                    fps=1000 / self.interval, dpi=200,
                    extra_args=["-vcodec", "libx264", "-preset", "slow", "-crf", "18"],
                )
                if self.verbose:
                    print("✅ Saved.")
            else:
                if self.verbose:
                    print("Live animation started. Close window to exit.")
                plt.show()

        except Exception as exc:
            print(f"Runtime error: {exc}")
        finally:
            # Only quit mixer here if pygame renderer didn't already tear everything down
            if self.audio and self.renderer != "pygame":
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeMoD Metronome — Sierpinski Triangle visualizer with metronome sync",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=25000)
    parser.add_argument("--batch", type=int, default=60)
    parser.add_argument("--color", type=str, default="#0066ff")
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vertices", nargs=6, type=float, default=None,
                        metavar=("x1", "y1", "x2", "y2", "x3", "y3"))
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--renderer", type=str, default="matplotlib",
                        choices=["matplotlib", "pygame"])
    parser.add_argument("--pure-metronome", action="store_true")
    parser.add_argument("--beep-frequency", type=int, default=440)

    args = parser.parse_args()

    custom_vertices = None
    if args.vertices:
        v = args.vertices
        custom_vertices = [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])]

    try:
        metro = DeMoDMetronome(
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
            beep_frequency=args.beep_frequency,
        )
        metro.run()
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        sys.exit(1)
