{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/22.11.tar.gz") {} }:

with pkgs;

mkShell {
  packages = [
    graphviz
    poetry
    python310.pkgs.venvShellHook
  ];
  venvDir = "./.VENV";
  postShellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:${cairo}/lib/:${zlib}/lib:$LD_LIBRARY_PATH
    poetry config virtualenvs.prefer-active-python true
    poetry install --all-extras
  '';
}
