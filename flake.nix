{
  description = "Development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }:
    utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python311;
      pythonPackages = python.pkgs;
    in {
      devShell = pkgs.mkShell {
        name = "ai-courses";
        nativeBuildInputs = [ pkgs.bashInteractive ];

        # Add anything in here if you want it to run when we run `nix develop`.
        buildInputs = with pythonPackages; [
          # Additional dev packages list here.
          setuptools
          wheel
          venvShellHook
          ipython
          jupyter
          jupyterlab
          numpy
        ];
        venvDir = ".venv";
        src = null;
        postVenv = ''
            unset SOURCE_DATE_EPOCH
        '';
        postShellHook = ''
            unset SOURCE_DATE_EPOCH
            unset LD_PRELOAD

            PYTHONPATH=$PWD/$venvDir/${python.sitePackages}:$PYTHONPATH
        '';
      };
    });
}