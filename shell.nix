{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/22.11.tar.gz") {} }:

with pkgs;

mkShell {
  venvDir = "./.VENV";
  buildInputs = [
    cairo
    graphviz
    poetry
    zlib
    python310.pkgs.venvShellHook
  ];
  postShellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:${cairo}/lib/:${zlib}/lib:$LD_LIBRARY_PATH
    poetry install -E fca -E doc
    spacy validate | grep en_core_web_sm || poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
  '';
}
