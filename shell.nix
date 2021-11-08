with import <nixpkgs> {}; (pkgs.buildFHSUserEnv { name = "fhs";

profile = ''
      set -e
      eval "$(micromamba shell hook -s bash)"
      micromamba activate base
      set +e
    '';



 }).env
