{
  description = "DeMoD Metronome — Sierpinski Triangle visualizer with interactive metronome features";

  inputs = {
    # Pin to a stable channel for reproducibility; update deliberately with `nix flake update`
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      # forEachSystem f  →  { x86_64-linux = f pkgs; aarch64-linux = f pkgs; … }
      forEachSystem = f: nixpkgs.lib.genAttrs supportedSystems
        (system: f (import nixpkgs { inherit system; }));

      pythonEnv = pkgs: pkgs.python312.withPackages (ps: with ps; [
        numpy
        matplotlib
        pygame
        scipy       # useful for DSP work alongside this project
      ]);
    in {
      # `nix develop`
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = [ (pythonEnv pkgs) pkgs.ffmpeg ];
          shellHook = ''
            echo ""
            echo "  DeMoD Metronome dev shell ready."
            echo "  Python 3.12 + NumPy, Matplotlib, Pygame, SciPy, FFmpeg"
            echo ""
            echo "  Quick start:"
            echo "    python demod_metronome.py --help"
            echo "    python demod_metronome.py --bpm 120 --interactive --renderer pygame"
            echo ""
          '';
        };
      });

      # `nix run`
      packages = forEachSystem (pkgs: {
        default = pkgs.writeShellApplication {
          name = "demod-metronome";
          runtimeInputs = [ (pythonEnv pkgs) pkgs.ffmpeg ];
          text = ''
            python ${./demod_metronome.py} "$@"
          '';
        };
      });
    };
}
