{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/20.09.tar.gz") {} }:

with pkgs;

  stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  venvDir = "./.VENV";
  buildInputs = [
    python38Full
    python38Packages.venvShellHook
    python38Packages.virtualenv

    #python38Packages.pandas
    python38Packages.cairocffi
    python38Packages.spacy
    python38Packages.toolz

    python38Packages.jupyter
    python38Packages.matplotlib
  ];
  postShellHook = ''
    export PS1="\$(__git_ps1) $PS1"
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm-3.0.0
    pip install -e ".[dev,doc]"
  '';
}
