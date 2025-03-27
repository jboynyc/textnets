{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.11.tar.gz") {} }:

with pkgs;

mkShell {
  packages = [
    graphviz
    pdm
    python312Packages.venvShellHook
  ];
  venvDir = "./.venv";
  postShellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:${zlib}/lib:$LD_LIBRARY_PATH
    pdm install -d
  '';
}
