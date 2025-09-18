{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/25.05.tar.gz") {} }:

with pkgs;

mkShell rec {
  packages = [
    autoPatchelfHook
    pdm
    python312Packages.venvShellHook
  ];
  venvDir = "./.venv";
  LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH";  # needed for numpy
  postShellHook = ''
    autoPatchelf ${venvDir}/bin > /dev/null  # needed for ruff
    pdm install -qd
  '';
}
