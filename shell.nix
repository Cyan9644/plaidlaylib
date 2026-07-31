{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.liburing
    # `make format` runs Google-style clang-format over the repo's sources.
    pkgs.clang-tools
    # `make bench` drives the benchmarks and plots them with this Python.
    (pkgs.python3.withPackages (ps: [ ps.matplotlib ps.numpy ]))
  ];
}
