{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.wheel
  ];
}
