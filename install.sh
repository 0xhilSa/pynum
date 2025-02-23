#!/bin/bash

if python3 -c "import pynum" 2>/dev/null; then
  read -p "pynum is already installed. Do you want to reinstall it? (y/N): " choice
  case "$choice" in
    y|Y)
      echo "Reinstalling pynum..."
      pip uninstall -y pynum
      ;;
    *)
      echo "Aborting installation."
      exit 1
      ;;
  esac
fi

python3 -m build
cd dist/ && pip install *.whl && cd ..
rm -rf dist/ build/ pynum.egg-info/
