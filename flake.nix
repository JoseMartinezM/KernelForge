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
          baseNativeLibs = pkgs: builtins.attrValues {
            inherit (pkgs)
              zlib
              zstd
              blas
              lapack
              ;

            libstd = pkgs.stdenv.cc.cc.lib;
          };

          basePackages = pkgs: [
            pkgs.uv
            pkgs.nodejs-slim_22
          ];
        in
        {
          default = let
            pkgs = import nixpkgs { inherit system; };
          in pkgs.mkShell {
            packages = basePackages pkgs;

            shellHook = ''
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              source .venv/bin/activate
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (baseNativeLibs pkgs)}:$LD_LIBRARY_PATH"
            '';
          };

          rocm = let
            pkgs = import nixpkgs {
              inherit system;
              config.rocmSupport = true;
            };
          in pkgs.mkShell {
            packages = basePackages pkgs ++ [
              pkgs.rocmPackages.rocminfo
              pkgs.rocmPackages.clr
              pkgs.rocmPackages.llvm.clang
            ];

            shellHook = ''
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              source .venv/bin/activate
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (baseNativeLibs pkgs)}:$LD_LIBRARY_PATH"
            '';
          };
        });
    };
}
