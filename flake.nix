{
  description = "KernelForge development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };

          nativeLibs = builtins.attrValues {
            inherit (pkgs)
              zlib
              blas
              lapack
              ;

            libstd = pkgs.stdenv.cc.cc.lib;
          };
        in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.python3 pkgs.uv ];

            shellHook = ''
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              source .venv/bin/activate
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}:$LD_LIBRARY_PATH"
            '';
          };
        });
    };
}
