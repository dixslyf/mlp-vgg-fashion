{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages = [
              pkgs.just
              (pkgs.python312.withPackages (
                ps: with ps; [
                  jupyterlab
                  jupytext
                ]
              ))
              pkgs.quarto
            ];
          };
        };
      }
    );
}
