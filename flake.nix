{
  description = "DeMoD Metronome - A Python-based Sierpinski Triangle visualization with interactive metronome features";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f { 
        pkgs = import nixpkgs { inherit system; };
      });
    in
    forEachSupportedSystem ({ pkgs }: {
      devShells.default = pkgs.mkShell {
        packages = [
          (pkgs.python312.withPackages (ps: with ps; [
            numpy
            matplotlib
            pygame
          ]))
          pkgs.ffmpeg  # For saving MP4 videos
        ];
        shellHook = ''
          echo "Welcome to the DeMoD Metronome dev shell!"
          echo "Python with NumPy, Matplotlib, Pygame, and FFmpeg is ready."
          echo "Run: python demod_metronome.py --help"
        '';
      };
    });
}