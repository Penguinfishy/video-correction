#!/bin/bash
# Creates "Band Stabilizer.app" on your Desktop that launches this folder's app.py.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$HOME/Desktop/Band Stabilizer.app"
if [ ! -x "$DIR/.venv/bin/python" ]; then
  echo "Run the setup steps first (.venv not found in $DIR)"; exit 1
fi
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cat > "$APP/Contents/MacOS/BandStabilizer" <<EOF
#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "$DIR"
exec "$DIR/.venv/bin/python" "$DIR/app.py"
EOF
chmod +x "$APP/Contents/MacOS/BandStabilizer"
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Band Stabilizer</string>
  <key>CFBundleExecutable</key><string>BandStabilizer</string>
  <key>CFBundleIdentifier</key><string>local.band-stabilizer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
echo "Created: $APP"
