{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              pkgs.glibcLocales
              pkgs.python3
              pkgs.uv
            ];

            # LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
            # LANG = "ru_RU.UTF-8";
            # LC_ALL = "ru_RU.UTF-8";

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
