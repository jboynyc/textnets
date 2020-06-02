with import <nixpkgs> {};

  stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  venvDir = "./.VENV";
  buildInputs = [
    python38Full
    python38Packages.venvShellHook
    python38Packages.virtualenv

    python38Packages.click
    python38Packages.pandas
    python38Packages.pycairo
    python38Packages.spacy
    python38Packages.toolz
  ];
  postShellHook = ''
    pip install -e ".[dev,doc]"
  '';
}
