#!/bin/sh
# Renders each scene to images/<scene>.png at 1920x1080 with headless Chrome (fonts and rough.js load from CDNs).
cd "$(dirname "$0")"
for js in ${@:-[0-9]*.js}; do
  name="${js%.js}"
  sed "s/SCENE/$js/" frame.html > "$name.html"
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
    --window-size=1920,1080 --virtual-time-budget=8000 --screenshot="../images/$name.png" "file://$PWD/$name.html" 2>/dev/null
  rm "$name.html"
  echo "rendered images/$name.png"
done
