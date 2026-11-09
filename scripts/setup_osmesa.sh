#!/usr/bin/env bash
# Isolated software renderer for Cannon's MIG GPUs; no system/Python package changes.
set -euo pipefail
source "$(dirname "$0")/common.sh"
target="$DQAT_ROOT/data/osmesa-runtime"
package=mesa-libOSMesa-23.1.4-4.el8_10.x86_64.rpm
mkdir -p "$target/packages"
cd "$target"
if [[ ! -f "packages/$package" ]]; then
  curl --fail --location --max-time 90 --silent --show-error \
    "https://download.rockylinux.org/pub/rocky/8/AppStream/x86_64/os/Packages/m/$package" \
    --output "packages/$package.partial"
  mv "packages/$package.partial" "packages/$package"
fi
echo "e3e85579f7987611fb01c8d805537265607d1cd6a2eb58f6e60c7fc222d3ddd3  packages/$package" | sha256sum --check
rpmkeys --checksig "packages/$package"
if [[ ! -f usr/lib64/libOSMesa.so.8.0.0 ]]; then
  rpm2cpio "packages/$package" | cpio -idm --quiet
fi
echo '8e8ad6bc3f6d443555d495505e8f2b2f8a07fcfef27cf0f7538ad2c82b328f1c  usr/lib64/libOSMesa.so.8.0.0' | sha256sum --check
