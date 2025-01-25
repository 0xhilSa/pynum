#!/bin/sh

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
PKG="pynum"

# Define directories and files to check
DIRS=("/dist" "/build" "/pynum.egg-info")
SHARED_FILES=("cuda_stream.cpython-310-x86_64-linux-gnu.so")
SETUP_FILE="./setup.py"

# Clear directories if they exist
echo -e "${CYAN}Checking and cleaning directories...${NC}"
for DIR in "${DIRS[@]}"; do
  if [ -d "$DIR" ]; then
    echo -e "${GREEN}Removing existing directory: $DIR${NC}"
    rm -rf "$DIR"
  else
    echo -e "${YELLOW}Directory not found: $DIR${NC}"
  fi
done

# Check for shared files
echo -e "${CYAN}Checking for required shared files...${NC}"
for SHARED_FILE in "${SHARED_FILES[@]}"; do
  if [ ! -f "./pynum/src/$SHARED_FILE" ]; then
    echo -e "${RED}Error: Required shared file not found: $SHARED_FILE${NC}"
    exit 1
  else
    echo -e "${GREEN}Found shared file: $SHARED_FILE${NC}"
  fi
done

# Check for setup.py file
echo -e "${CYAN}Checking for setup.py...${NC}"
if [ ! -f "$SETUP_FILE" ]; then
  echo -e "${RED}Error: setup.py file not found!${NC}"
  exit 1
else
  echo -e "${GREEN}Found setup.py${NC}"
fi

# Build, uninstall, and reinstall the package
echo -e "${CYAN}Building the Python package...${NC}"
python3 setup.py sdist bdist_wheel

echo -e "${CYAN}Uninstalling the existing pynum package (if any)...${NC}"
pip uninstall -y pynum

echo -e "${CYAN}Installing the new package...${NC}"
pip install dist/*whl

echo -e "${GREEN}${PKG} installed successfully.${NC}"
