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
      pkgs = import nixpkgs {
        inherit system;
      };
    in {
      devShell = pkgs.mkShell {
        # Add anything in here if you want it to run when we run `nix develop`.
        buildInputs = with pkgs; [
          # Additional dev packages list here.
          python3
        ];
      };
    });
}
