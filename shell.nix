{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/23.11.tar.gz") {} }:

with pkgs;

mkShell {
  packages = [
    graphviz
    poetry
    python311.pkgs.venvShellHook
  ];
  venvDir = "./.VENV";
  postShellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:${cairo}/lib/:${zlib}/lib:$LD_LIBRARY_PATH
    poetry config virtualenvs.prefer-active-python true
    poetry install --with doc --all-extras
  '';
}
