{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.liburing
    # `make bench` drives the benchmarks and plots them with this Python.
    (pkgs.python3.withPackages (ps: [ ps.matplotlib ps.numpy ]))
  ];
}
